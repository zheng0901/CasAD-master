# Environment
```
# create virtual environment
conda create --name casad python=3.9

# activate environment
conda activate casad

# install pytorch==1.12.0
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

# install other requirements
pip install -r requirements.txt
```
# Run the Code 
- An example from a Weibo dataset with a 0.5 hour observation time.
```
# generate information cascades
python gen_cas.py --input=./dataset/weibo/ --observation_time=1800 --prediction_time=86400

# generate cascade graph and global graph embeddings 
python gen_emb.py --input=./dataset/weibo/ --observation_time=1800

# run CasAD model
python model.py --input=./dataset/weibo/ 
```
You can modify the options in the running code accordingly based on the dataset, the corresponding observation time, and prediction time, as shown in the table below: 
### Observation Time

| Dataset | Setting 1 | # in Code | Setting 2 | # in Code |
|:--------|:----------|:----------|:----------|:----------|
| Twtter  | 1 day     | 86400     | 2 days    | 172800    |
| Weibo   | 0.5 hour  | 1800      | 1 hour    | 3600      |
| APS     | 3 years   | 1095      | 5 years   | 1826      |

### Prediction Time

| Dataset | Time     | # in Code |
|:--------|:---------|:----------|
| Twitter | 32 days  | 2764800   |
| Weibo   | 24 hours | 86400     |
| APS     | 20 years | 7305      |
