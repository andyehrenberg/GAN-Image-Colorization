To train with discriminator using batch normalization:

python3 train.py --max_epoch=20 --smoothing=0.9 --l1_weight=0.99 --base_lr_gen=3e-4
--base_lr_disc=6e-5 --lr_decay_steps=6e4 --disc_norm=batch --apply_weight_init=1

To train with discriminator using spectral normalization:

python3 train.py --max_epoch=20 --smoothing=1.0 --l1_weight=0.985 --base_lr_gen=2e-4
--base_lr_disc=2e-4 --lr_decay_steps=4e4 --disc_norm=spectral --apply_weight_init=0


To test models (need to change paths to files with weights of generators):

python3 test.py
