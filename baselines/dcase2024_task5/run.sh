echo "Train the baseline system"
for seed in 1234 2345 3456 4567 5678; do
    python3 train.py seed=${seed} exp_name="baseline2024-seed${seed}"
done


echo "Compare different features"
for i in 1 2 3 4 5; do
    python3 train.py features.feature_types="pcen" exp_name="pcen"
    python3 train.py features.feature_types="pcen@mfcc" exp_name="pcen@mfcc"
    python3 train.py features.feature_types="mel" exp_name="mel"
    python3 train.py features.feature_types="logmel@mfcc" exp_name="logmel@mfcc"
    python3 train.py features.feature_types="logmel@delta_mfcc" exp_name="logmel@delta_mfcc"
    python3 train.py features.feature_types="pcen@delta_mfcc" exp_name="pcen@delta_mfcc"
done


echo "Train the baseline system w/o negative contrastive"
for seed in 1234 2345 3456 4567 5678; do
    python3 train.py train_param.negative_train_contrast=false seed=${seed} exp_name="baseline2024-no_neg_contrast-seed${seed}"
done