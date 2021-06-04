#51

#python pre_processing.py data/final/target2 5 connect resize
for i in {1..13}
do
    python  main.py             data/final/target2/classification/t$i 100000 0.0005 sift p
    cp      results/result.png  data/final/target2/results1/t$i.png
done

#python pre_processing.py data/final/target1/results1 2 NOconnect NOresize
#for i in {1..5}
#do
#    python  main.py             data/final/target2/results1/classification/t$i 40000 0.01 sift p
#    cp      results/result.png  data/final/target2/results1/results2/t$i.png
#done

#python pre_processing.py data/final/target1/results1/results2 2 NOconnect NOresize
#for i in {1..2}
#do
#    python  main.py             data/final/target1/results1/results2/classification/t$i 80000 0.01 sift a
#    cp      results/result.png  data/final/target1/results1/results2/results3/t$i.png
#done

#python  main.py             data/final/target1/results1/results2/results3 80000 0.01 sift p


#python pre_processing.py data/final/target1/results1/results2/results3 2 NOconnect NOresize
#for i in 2
#do
#    python  main.py             data/final/target1/results1/results2/results3/classification/t$i 15000 0.3 orb a
#    cp      results/result.png  data/final/target1/results1/results2/results3/results4/t$i.png
#done

#python  main.py             data/final/target1/results1/results2/results3/results4 100000 0.03 sift p
#cp      results/result.png  data/final/target1/results1/results2/results3/results4/result.png
