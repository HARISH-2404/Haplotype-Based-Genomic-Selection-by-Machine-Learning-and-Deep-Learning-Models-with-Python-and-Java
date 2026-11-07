#!/usr/bin/env bash
# Compile without Maven (works with any JDK >= 11). Output classes go to ./out
set -e
cd "$(dirname "$0")"
mkdir -p out
if command -v javac >/dev/null 2>&1; then JAVAC="javac"; else JAVAC="java -m jdk.compiler/com.sun.tools.javac.Main"; fi
$JAVAC -d out src/main/java/org/hgs/*.java
echo "compiled -> $(pwd)/out"
