#!/bin/bash
# Daily vault-agent pipeline
# Processes notes, normalizes tags, validates links, and applies everything.

set -euo pipefail

LOG_DIR="$HOME/.vault-agent/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y-%m-%d).log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== vault-agent daily run ==="

# Ensure Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    log "Starting Ollama..."
    open -a Ollama
    sleep 10
fi

# Step 1: Process new/changed notes
log "Step 1: Processing notes..."
vault-agent run 2>&1 | tee -a "$LOG_FILE"

# Step 2: Normalize tags
log "Step 2: Normalizing tags..."
vault-agent normalize-tags --apply 2>&1 | tee -a "$LOG_FILE"

# Step 3: Validate and fix links
log "Step 3: Validating links..."
vault-agent validate --fix 2>&1 | tee -a "$LOG_FILE"

# Step 4: Clean duplicates
log "Step 4: Cleaning links..."
vault-agent clean-links 2>&1 | tee -a "$LOG_FILE"

# Step 5: Discover links via shared tags
log "Step 5: Discovering links..."
vault-agent discover-links --max-tag-freq 10 2>&1 | tee -a "$LOG_FILE"

# Step 6: Apply everything
log "Step 6: Applying tags and links..."
vault-agent apply-tags -y 2>&1 | tee -a "$LOG_FILE"
vault-agent apply-links -y 2>&1 | tee -a "$LOG_FILE"

# Final status
log "Step 7: Final status"
vault-agent status 2>&1 | tee -a "$LOG_FILE"

log "=== Done ==="
