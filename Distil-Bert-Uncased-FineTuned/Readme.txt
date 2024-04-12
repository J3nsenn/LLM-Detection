There are 2 jupiter notebooks
1. "Training-code-with-metrics" was the original code used to train the model and upload it to the hugging face under the huygging face repositry: "fine-tuned-distilbertv3"
This is not essential to run and will take in excess of 6 hrs on a CPU. 

2. "test-perf-distilbert" is the code used to download the trained model and assess its performance on the test dataset. This will take about 1 hour on a cpu

3. In both notebooks, dataset is a very finnicky library that is difficult to work with. "Correct_Loaddataset_output.png" should be
displayed when run as an indicator in BOTH notebooks that the dataformat has loaded correctly for the test, train or val set to be taken in by the model. 

4. Some lines of code have been intentionlly left commented out with explantions in markdown or in comments
