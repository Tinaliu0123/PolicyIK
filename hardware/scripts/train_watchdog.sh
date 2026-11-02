#!/usr/bin/env bash
# Check every checkpoint of a training run; on a fault, stop the trainer (it saves first) and move the arm home.
#
#   bash scripts/train_watchdog.sh <run_dir> [poll_s]           # same machine as the trainer
#   WATCHDOG_NO_HOME=1 bash scripts/train_watchdog.sh <run_dir>  # leave the arm where it is
set -u
mkdir -p "$1"; RUN=$(cd "$1" && pwd); POLL=${2:-30}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOG=$RUN/watchdog.log
NAME=$(basename "$RUN")
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
# anchored, so a shell line that only contains the text is never signalled
trainer_pid() { pgrep -f "^python3 -m policyik_hw\.train .*--name $NAME( |$)" | tr '\n' ' '; }
say "watchdog on $RUN (poll ${POLL}s): checks every new buffer_step*, stops the trainer and parks the arm on ABNORMAL"
seen=""
for i in $(seq 1 18); do [ -n "$(trainer_pid)" ] && break; sleep 10; done
[ -n "$(trainer_pid)" ] || say "no trainer process found for --name $NAME yet; watching the folder anyway"
while true; do
    for b in $(ls -tr "$RUN"/buffer_step* 2>/dev/null | grep -v '\.her$'); do
        step=${b##*buffer_step}
        case " $seen " in *" $step "*) continue;; esac
        [ -f "$b.her" ] || continue
        s1=$(stat -c %s "$b"); sleep 5; s2=$(stat -c %s "$b"); [ "$s1" = "$s2" ] || continue
        seen="$seen $step"
        say "checkpoint step $step saved: checking the last 5000 rows"
        out=$(cd "$ROOT" && python3 -m policyik_hw.watchdog_check "$RUN" "$step" 2>&1); rc=$?
        echo "$out" >> "$LOG"
        if [ $rc -eq 2 ]; then
            say "ABNORMAL at step $step -> stopping the trainer (Ctrl-C path: it saves first)"
            echo "$out" | grep -A20 'ABNORMAL' | tee -a "$LOG"
            pids=$(trainer_pid)
            if [ -n "$pids" ]; then
                say "trainer pid(s): $pids"
                for p in $pids; do kill -INT "$p"; done
                for i in $(seq 1 60); do [ -z "$(trainer_pid)" ] && break; sleep 2; done
                if [ -n "$(trainer_pid)" ]; then say "trainer still alive after 120 s: second Ctrl-C"; for p in $(trainer_pid); do kill -INT "$p"; done; sleep 10; fi
                [ -n "$(trainer_pid)" ] && { say "trainer did not exit; NOT parking the arm (two publishers)"; exit 2; }
                say "trainer exited"
            else
                say "no trainer process found"
            fi
            if [ "${WATCHDOG_NO_HOME:-0}" = 1 ]; then say "WATCHDOG_NO_HOME set: arm left where it is"; exit 2; fi
            say "parking the arm"; (cd "$ROOT" && python3 -m policyik_hw.robot.park 2>&1 | tee -a "$LOG")
            say "stopped. Last checkpoint BEFORE this one is the clean resume point (see the checks above)."
            exit 2
        elif [ $rc -eq 0 ]; then
            say "step $step: normal, training continues"
        else
            say "step $step: could not check (rc $rc), training continues"
        fi
    done
    if [ -n "$(trainer_pid)" ]; then sleep "$POLL"; else
        sleep 15; [ -n "$(trainer_pid)" ] && continue
        if [ -f "$RUN/buffer" ] && [ -f "$RUN/summary.json" ]; then
            say "trainer finished: checking the final buffer"; (cd "$ROOT" && python3 -m policyik_hw.watchdog_check "$RUN" final 2>&1 | tee -a "$LOG")
        fi
        say "trainer exited; watchdog done"; exit 0
    fi
done
