#!/bin/bash
# Shared helpers for the deploy scripts (deploy.sh, start.sh, stop.sh).
#
# load_env   reads the configuration file (default $SCRIPT_DIR/.env, override
#            with CONFIG_ENV) with per-key precedence:
#              shell environment  >  config file  >  script defaults
#            A key already exported in the shell wins over the file; a key
#            still unset after load_env() falls back to ${VAR:-default} in the
#            calling script. Values may be bare, "double-quoted" or
#            'single-quoted'; comments (#) and blank lines are ignored.
#            No variable expansion happens inside the file.
#
# join_remote  quotes argv into a single %q-escaped string that survives a
#              remote shell round-trip (flags containing spaces stay intact).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

load_env() {
  local env_file="${CONFIG_ENV:-$SCRIPT_DIR/.env}"
  [ -f "$env_file" ] || return 0
  local line key val
  while IFS='=' read -r key val; do
    case "$key" in
      '' | \#*) continue ;;
    esac
    key="${key%"${key##*[![:space:]]}"}" # trim trailing whitespace
    case "$key" in
      [A-Za-z_]*[A-Za-z0-9_]*) ;;        # valid identifier
      *) continue ;;
    esac
    [ -n "${!key:-}" ] && continue       # shell env wins
    val="${val%"${val##*[![:space:]]}"}" # trim trailing whitespace
    case "$val" in
      \"*\") val="${val:1:${#val}-2}" ;;
      \'*\') val="${val:1:${#val}-2}" ;;
    esac
    export "$key=$val"
  done < "$env_file"
}

join_remote() {
  local e out=()
  for e in "$@"; do
    [ -n "$e" ] || continue
    out+=("$(printf '%q' "$e")")
  done
  printf '%s' "${out[*]}"
}
