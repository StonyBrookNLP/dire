#!/bin/bash

set -e
set -x

pip install gdown
# If gdown doesn't work, you can download files from mentioned URLs manually
# and put them at appropriate locations.

# URL: https://drive.google.com/file/d/1NbTw2ncC07IzTkWcw9743J0EYEESpzE4/view?usp=sharing
gdown --id 1NbTw2ncC07IzTkWcw9743J0EYEESpzE4 --output data/processed/original_hotpotqa_dev.json

# URL: https://drive.google.com/file/d/18p6N7LZEuNQxURqi9uB0BNWnBMmXAtjY/view?usp=sharing
gdown --id 18p6N7LZEuNQxURqi9uB0BNWnBMmXAtjY --output data/processed/original_hotpotqa_train.json


# URL: https://drive.google.com/file/d/1XeGbZLUj_XCQoDZx9t4emYBFsjJOOorg/view?usp=sharing
gdown --id 1XeGbZLUj_XCQoDZx9t4emYBFsjJOOorg --output data/processed/probe_of_original_hotpotqa_dev.json

# URL: https://drive.google.com/file/d/1fjjpowU2iM7_bLIzIOiAfsmR7xlVUaUe/view?usp=sharing
gdown --id 1fjjpowU2iM7_bLIzIOiAfsmR7xlVUaUe --output data/processed/probe_of_original_hotpotqa_train.json


# URL: https://drive.google.com/file/d/14em2OnOUqLYlJENHIy4WdAB6Zbvte28Q/view?usp=sharing
gdown --id 14em2OnOUqLYlJENHIy4WdAB6Zbvte28Q --output data/processed/transformed_hotpotqa_dev.json

# URL: https://drive.google.com/file/d/1F_8bu7zFRjJMV5NReeW9LJR64NxpDdOx/view?usp=sharing
gdown --id 1F_8bu7zFRjJMV5NReeW9LJR64NxpDdOx --output data/processed/transformed_hotpotqa_train.json


# URL: https://drive.google.com/file/d/1mP7e23IqueCXYTlkC_LaIQWOyUHSf4yl/view?usp=sharing
gdown --id 1mP7e23IqueCXYTlkC_LaIQWOyUHSf4yl --output data/processed/probe_of_transformed_hotpotqa_dev.json

# URL: https://drive.google.com/file/d/1I8FVcVnGH8KnQEiB2sib8RzbRi4UcMcu/view?usp=sharing
gdown --id 1I8FVcVnGH8KnQEiB2sib8RzbRi4UcMcu --output data/processed/probe_of_transformed_hotpotqa_train.json

