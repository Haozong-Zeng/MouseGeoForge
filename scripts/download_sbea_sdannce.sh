#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
selection="${1:-all}"

wget_common=(
  --continue
  --tries=0
  --retry-connrefused
  --timeout=30
  --read-timeout=60
  --waitretry=5
  --user-agent=Mozilla/5.0
  --progress=dot:giga
)

download_checked() {
  local url="$1"
  local destination="$2"
  local expected_size="$3"
  local expected_md5="$4"
  local partial="${destination}.part"

  if [[ -f "$destination" ]]; then
    local current_size
    current_size="$(stat -c %s "$destination")"
    if [[ "$current_size" == "$expected_size" ]] && \
       [[ "$(md5sum "$destination" | awk '{print $1}')" == "$expected_md5" ]]; then
      echo "verified existing: $destination"
      return 0
    fi
    echo "existing file failed verification: $destination" >&2
    return 1
  fi

  mkdir -p "$(dirname "$destination")"
  wget "${wget_common[@]}" --output-document "$partial" "$url"

  local downloaded_size
  downloaded_size="$(stat -c %s "$partial")"
  if [[ "$downloaded_size" != "$expected_size" ]]; then
    echo "size mismatch for $partial: got $downloaded_size expected $expected_size" >&2
    return 1
  fi

  local downloaded_md5
  downloaded_md5="$(md5sum "$partial" | awk '{print $1}')"
  if [[ "$downloaded_md5" != "$expected_md5" ]]; then
    echo "md5 mismatch for $partial: got $downloaded_md5 expected $expected_md5" >&2
    return 1
  fi

  mv "$partial" "$destination"
  echo "downloaded and verified: $destination"
}

download_sbea() {
  local target="$repo_root/dataset/SBeA/raw/SBeA_upload_data.zip"
  download_checked \
    "https://ndownloader.figshare.com/files/39698587" \
    "$target" \
    "17499141394" \
    "3b753b4a00accf596b349bbc9d4b3a44"
}

download_sdannce_mouse() {
  local root="$repo_root/dataset/sDANNCE-Mouse"
  local raw="$root/raw"
  local metadata="$root/dataverse_dataset.json"
  mkdir -p "$raw"

  wget "${wget_common[@]}" \
    --output-document "${metadata}.part" \
    "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/VKJHTD"
  mv "${metadata}.part" "$metadata"

  python3 - "$metadata" <<'PY' | while IFS=$'\t' read -r file_id filename expected_size expected_md5; do
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    dataset = json.load(handle)

for entry in dataset["data"]["latestVersion"]["files"]:
    data_file = entry["dataFile"]
    checksum = data_file["checksum"]
    if checksum["type"].upper() != "MD5":
        raise RuntimeError(f"Unsupported checksum type: {checksum['type']}")
    print(
        data_file["id"],
        data_file["filename"],
        data_file["filesize"],
        checksum["value"].lower(),
        sep="\t",
    )
PY
    download_checked \
      "https://dataverse.harvard.edu/api/access/datafile/${file_id}" \
      "$raw/$filename" \
      "$expected_size" \
      "$expected_md5"
  done

  download_checked \
    "https://dataverse.harvard.edu/api/access/datafile/10946391" \
    "$root/README_SOURCE.txt" \
    "3037" \
    "44c02394bd2db775b09a98e435ec2657"
  download_checked \
    "https://dataverse.harvard.edu/api/access/datafile/10945217" \
    "$root/sDANNCE_file_info.xlsx" \
    "52330" \
    "5cbde4fd5a476c27102568f2918159d1"
}

case "$selection" in
  sbea)
    download_sbea
    ;;
  sdannce)
    download_sdannce_mouse
    ;;
  all)
    download_sbea
    download_sdannce_mouse
    ;;
  *)
    echo "usage: $0 [all|sbea|sdannce]" >&2
    exit 2
    ;;
esac
