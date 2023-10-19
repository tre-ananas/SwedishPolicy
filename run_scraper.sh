#!/bin/bash

mkdir -p Logs

start=1
increment=3
end=10

while [[ $start -lt $end ]]; do
    next=$((start + increment))
    output_file="Logs/output_${start}_${next}"
    
    ~/chsres/SwedishPolicy/GovDocsOperationalV3.py $start $next > "${output_file}_out.txt" 2> "${output_file}_err.txt" &

    start=$next
done
