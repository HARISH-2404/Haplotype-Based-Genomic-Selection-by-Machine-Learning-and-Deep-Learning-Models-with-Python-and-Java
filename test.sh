#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
./build.sh
mkdir -p out-test
if command -v javac >/dev/null 2>&1; then JAVAC="javac"; else JAVAC="java -m jdk.compiler/com.sun.tools.javac.Main"; fi
$JAVAC -cp out -d out-test src/test/java/org/hgs/SelfTest.java
java -cp out:out-test org.hgs.SelfTest
