#!/bin/bash
# long match as short processes: ./chunks.sh <chunks> <pairs per chunk>  (env as for match.js); prints the running total after every chunk
CH=${1:-8}; PP=${2:-300}; W=0; N=0; PW=0; PL=0; TL=0
for ((i = 0; i < CH; i++)); do
  L=$(OFFSET=$((i * PP)) node match.js $PP ${WORKERS:-8} | tail -1)
  read w n < <(echo "$L" | sed -E 's|new vs old: ([0-9]+)/([0-9]+).*|\1 \2|')
  pw=$(echo "$L" | sed -E 's/.*pairs won ([0-9]+) lost ([0-9]+).*/\1/'); pl=$(echo "$L" | sed -E 's/.*pairs won ([0-9]+) lost ([0-9]+).*/\2/')
  tl=$(echo "$L" | sed -E 's/.*turn limit ([0-9]+).*/\1/')
  W=$((W + w)); N=$((N + n)); PW=$((PW + pw)); PL=$((PL + pl)); TL=$((TL + tl))
  awk -v w=$W -v n=$N -v pw=$PW -v pl=$PL -v tl=$TL -v i=$((i + 1)) -v c=$CH 'function e(x){return 400*log(x/(1-x))/log(10)} BEGIN{p=w/n; se=sqrt(p*(1-p)/n); printf "chunk %d/%d total %d/%d = %.1f%% ±%.1f  Elo %.0f (%.0f..%.0f)  turn limit %d, pairs won %d lost %d\n", i,c,w,n,100*p,196*se,e(p),e(p-1.96*se),e(p+1.96*se),tl,pw,pl}'
done
