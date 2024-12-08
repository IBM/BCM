BENCHMARK=$1
SHOT=$2
METHOD=$3
GPU=$4
FILENAME=$5

if [ ${BENCHMARK} == "pascal5i" ]
then
  DATA="pascal"
  SPLITS="0 1 2 3"
elif [ ${BENCHMARK} == "coco20i" ]
then
  DATA="coco"
  SPLITS="0 1 2 3"
elif [ ${BENCHMARK} == "pascal10i" ]
then
  DATA="pascal"
  SPLITS="10 11"
fi

printf "%s\nbenchmark: ${BENCHMARK}, shot: ${SHOT}, method: ${METHOD}\n\n" "---" >> ${FILENAME}
for SPLIT in $SPLITS
do
  python3 -m src.test --config config/${DATA}.yaml \
            --opts split ${SPLIT} \
              shot ${SHOT} \
              pi_estimation_strategy self \
              n_runs 5 \
              gpus ${GPU} \
              method ${METHOD} \
              beta 0.5 \
              ensemble True \
              sampling "bg" \
              top_k 1 \
              |& tee -a ${FILENAME}
  printf "\n" >> ${FILENAME}
done
