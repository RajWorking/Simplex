for i in {0..9}
do
    filename=cases/inp-$i.txt
    echo $filename
    sed '1,/^$/{/^$/!d}' $filename
    python 1.py < $filename
    echo "------------------------------------"
done