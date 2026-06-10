isic
只跑Setting I 的 fold 1
SETTINGS="1" FOLDS1="1" GPUID1=0 bash scripts/train_isic_all.sh
SETTINGS="1" FOLDS1="1" GPUID1=0 bash scripts/eval_isic_all.sh

NSTEP=57000 SETTINGS="1" FOLDS1="1" GPUID1=0 bash scripts/train_isic_all.sh
保存：snapshots/39000.pth
find exps_train_on_isic_setting1_FSMIS_FoB -path "*cv1/*/snapshots/39000.pth" | sort -V
CKPT_STEP=39000 SETTINGS="1" FOLDS1="1" GPUID1=0 bash scripts/eval_isic_all.sh
如果 fold 1 的结果明显上来了，再继续训练剩下的 folds：
NSTEP=57000 SETTINGS="1" FOLDS1="2 3 4 5" GPUID1=0 bash scripts/train_isic_all.sh
CKPT_STEP=39000 SETTINGS="1" GPUID1=0 bash scripts/eval_isic_all.sh
grep -R "Category .* Dice\|Mean Dice" -n logs_isic_eval | grep setting1

跑完整 Setting I
SETTINGS="1" GPUID1=0 bash scripts/train_isic_all.sh
SETTINGS="1" GPUID1=0 bash scripts/eval_isic_all.sh

再跑 Setting II
SETTINGS="2" GPUID1=0 bash scripts/train_isic_all.sh
SETTINGS="2" GPUID1=0 bash scripts/eval_isic_all.sh

两个 setting 都跑
SETTINGS="1 2" GPUID1=0 bash scripts/train_isic_all.sh
SETTINGS="1 2" GPUID1=0 bash scripts/eval_isic_all.sh