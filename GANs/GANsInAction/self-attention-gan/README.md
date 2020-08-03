# Self-Attention GAN

Emplement [Self-Attention GAN](https://arxiv.org/abs/1805.08318) in Pytorch

## Prerequiresites
 * Python 3.5+
 * PyTorch

<p align="center"><img width="100%" src="assets/main_model.png" /></p>


## Usage
### 1. Clone the repository
``` bash
$ git clone
$ cd 
```

### 2. Install datasets (CelebA or LSUN)
``` bash
$ bash download.sh CelebA
or
$ bash download.sh LSUN
```

### 3. Train
```bash
$ python python main.py --batch_size 64 --imsize 64 --dataset celeb --adv_loss hinge --version sagan_celeb
or
$ python python main.py --batch_size 64 --imsize 64 --dataset lsun --adv_loss hinge --version sagan_lsun
```

### 4. Test
```
bash
$ cd samples/sagan_celeb
or
$ cd samples/sagan_lsun
```