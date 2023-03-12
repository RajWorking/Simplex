for i in {0..9}
do
    filename=cases/1/inp-$i.txt
    echo $filename
    sed '1,/^$/{/^$/!d}' $filename
    python 1.py < $filename
    echo "------------------------------------"
done