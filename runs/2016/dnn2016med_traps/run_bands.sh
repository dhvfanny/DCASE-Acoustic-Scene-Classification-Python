p=0
for (( i=1 ; i<=39 ; i=i+1))
do
  cp -r traps$p traps$i 
  sed -i -e 's/traps'$p'/traps'$i'/g' traps$i/task1_scene_classification.yaml             
  sed -i -e 's/band: '$p'/band: '$i'/g' traps$i/task1_scene_classification.yaml
  p=$i  
done

for (( i=0 ; i<=39 ; i=i+1))
do
  nohup python traps$i/task1_scene_classification.py > traps$i/out.txt &
done

