## Base data folder
root_dir = ""

######## MediaEval ######## 

## Train and Test csvs
me15_train_csv_name = "mediaeval2015/train_file.txt"
me15_test_csv_name = "mediaeval2015/test_file.txt"

## Image graph data (Node Embeddings)
me15_image_vec_dir = "mediaeval2015_data/image_data/"

## Text graph data (Node Embeddings)
me15_text_vec_dir = "mediaeval2015_data/text_data/"


######## Weibo ########

## Train and Test csvs
we_train_csv_name = "weibo_dataset/train_file.txt"
we_test_csv_name = "weibo_dataset/test_file.txt"

## Image graph data (Node Embeddings)
we_image_vec_dir = "weibo_data/image_data/"

## Text graph data (Node Embeddings)
we_text_vec_dir = "weibo_data/text_data/"


######## Parameters ######## 
batch_size = 256
epochs = 40
lr = 1e-4
gradient_accumulation_steps = 2