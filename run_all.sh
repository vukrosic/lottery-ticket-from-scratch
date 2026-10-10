#!/bin/sh
# Runs the six steps, the tests and the exercise. Expected output: demo-output.txt
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
for f in step1_big.py step2_search.py step3_new_start.py step4_objections.py step5_signs.py step6_shortcuts.py tests.py exercise.py; do
	echo
	echo "=== $f ==="
	$PY $f
done
