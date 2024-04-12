There are 2 jupiter notebooks
1. "Training-code-with-metrics" was the original code used to train the model and upload it to the hugging face under the hugging face repositry: "fine-tuned-distilbertv3"
This is not essential to run and will take in excess of 6 hrs on a CPU. 

2. "test-perf-distilbert" is the code used to download the trained model and assess its performance on the test dataset. This will take about 1 hour on a cpu.

3. In both notebooks, an access token is required to download the datasets and trained model. In one of the code blocs, you will be asked to login to the huggingface hub enter the access token, and deselect any checkbox involving github.
Access token: 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

4. In both notebooks, dataset is a very finnicky library that is difficult to work with. "Correct_Loaddataset_output.png" is how the loaddataset bloc should look
when run as an indicator in BOTH notebooks that the dataformat has loaded correctly for the test, train or val set to be taken in by the model. Any errors can be addressed in the instructions
in markdown. 

5.. Some lines of code have been intentionlly left commented out with explantions in markdown or in comments. 
