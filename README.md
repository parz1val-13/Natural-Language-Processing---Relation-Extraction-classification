# Intro to NLP - Assignment 5

## Team
|Student name| CCID |
|------------|------|
|Shyam Varsani   |  varsani    |
|Kavit Gami   |   gami   |

Please note that CCID is **different** from your student number.

## TODOs

In this file you **must**:
- [x] Fill out the team table above. 
- [x] Make sure you submitted the URL on eClass.
- [x] Acknowledge all resources consulted (discussions, texts, urls, etc.) while working on an assignment.
- [x] Provide clear installation and execution instructions that TAs must follow to execute your code.
- [x] List where and why you used 3rd-party libraries.
- [x] Delete the line that doesn't apply to you in the Acknowledgement section.

## Acknowledgement 
In accordance with the UofA Code of Student Behaviour, we acknowledge that  
(**delete the line that doesn't apply to you**)

- We have listed all external resources we consulted for this assignment.

https://www.geeksforgeeks.org/bag-of-words-bow-model-in-nlp/

https://www.geeksforgeeks.org/removing-stop-words-nltk-python/?ref=lbp



 Non-detailed oral discussion with others is permitted as long as any such discussion is summarized and acknowledged by all parties.

## 3-rd Party Libraries
You do not need to list `nltk` and `pandas` here.

* `main.py L:[269-289]` used `[argparse]` for [implementing command line arguments].
* `main.py L:[98-135, 235-263]` used `[csv]` for [reading input file and creating output file].
* `main.py L:[152-153]` used `[re]` for [pre=processing data].
* `main.py L:[12-88]` used `[numpy]` for [implementing nbc algorithm].
* `main.py L:[12-88]` used `[math]` for [implementing nbc algorithm].

## Execution
Example usage: use the following command in the current directory.

`python3 src/main.py --train data/train.csv --test data/test.csv --output output/test.csv`

## Data

The assignment's training data can be found in [data/train.txt](data/train.txt),and the in-domain test data can be found in [data/test.txt](data/test.txt).
