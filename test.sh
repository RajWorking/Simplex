for i in {0..8}
do
    echo cases/inp-$i.txt
    python 1.py < cases/inp-$i.txt
    echo 
    cat cases/out-$i.txt
    echo
    echo "------------------------------------"
done