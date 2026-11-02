"""Read-only motor health monitor."""
import argparse, json, socket, struct, time
import numpy as np

ERR_NAMES = {0: "disabled", 1: "enabled", 8: "overvoltage", 9: "undervoltage",
             0xA: "overcurrent", 0xB: "MOS overtemp", 0xC: "rotor overtemp",
             0xD: "comms lost", 0xE: "overload"}
N_MOTORS = 8
FRAME_FMT = "=IB3x8s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="can1")
    ap.add_argument("--rate", type=float, default=20.0, help="publish Hz")
    ap.add_argument("--print-every", type=float, default=5.0, help="console table period, s (0 = off)")
    ap.add_argument("--no-ros", action="store_true", help="console only")
    a = ap.parse_args()

    s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    s.bind((a.iface,))
    s.settimeout(0.05)
    s.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                 b"".join(struct.pack("=II", 0x10 + i, 0x7FF) for i in range(1, N_MOTORS + 1)))

    pub = node = None
    if not a.no_ros:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
        rclpy.init()
        node = Node("motor_health")
        pub = node.create_publisher(String, "/motor_health", 10)

    err = np.full(N_MOTORS, -1, int); tmos = np.zeros(N_MOTORS); trot = np.zeros(N_MOTORS)
    seen = np.full(N_MOTORS, -1.0); n_rx = np.zeros(N_MOTORS, int)
    t_pub = t_print = time.monotonic()
    print(f"[motor_health] listening on {a.iface} (receive-only), publishing /motor_health at {a.rate:g} Hz")
    try:
        while True:
            try:
                pkt = s.recv(16)
                can_id, dlc, data = struct.unpack(FRAME_FMT, pkt)
                if dlc == 8 and 0x11 <= (can_id & 0x7FF) <= 0x10 + N_MOTORS:
                    j = (can_id & 0x7FF) - 0x11
                    err[j] = data[0] >> 4
                    tmos[j] = data[6]; trot[j] = data[7]
                    seen[j] = time.monotonic(); n_rx[j] += 1
            except socket.timeout:
                pass
            now = time.monotonic()
            if now - t_pub >= 1.0 / a.rate:
                t_pub = now
                age = np.where(seen < 0, 1e9, now - seen)
                msg = {"stamp": time.time(), "err": err.tolist(), "err_name": [ERR_NAMES.get(int(e), f"0x{int(e):X}") for e in err],
                       "tmos": tmos.tolist(), "trotor": trot.tolist(), "age_s": np.round(age, 3).tolist(), "n_rx": n_rx.tolist()}
                if pub is not None:
                    from std_msgs.msg import String
                    m = String(); m.data = json.dumps(msg); pub.publish(m)
            if a.print_every > 0 and now - t_print >= a.print_every:
                t_print = now
                age = np.where(seen < 0, 1e9, now - seen)
                row = "  ".join(f"J{i+1}:{ERR_NAMES.get(int(err[i]), '?')[:4]}/{tmos[i]:.0f}C"
                                + ("" if age[i] < 0.5 else " SILENT") for i in range(N_MOTORS))
                bad = [i + 1 for i in range(N_MOTORS) if err[i] not in (0, 1) or age[i] >= 0.5 or max(tmos[i], trot[i]) >= 60]
                print(f"[motor_health] {row}" + (f"   <-- ATTENTION J{bad}" if bad else ""), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        s.close()
        if node is not None:
            node.destroy_node()
            import rclpy; rclpy.shutdown()


if __name__ == "__main__":
    main()
