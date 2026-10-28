# Environmental Sound Classification with Deep Learning

## 1. Topic Decision

The selected topic is **environmental sound classification**. The project will investigate whether a pretrained audio representation can improve the classification of everyday sounds compared with a compact convolutional neural network (CNN) trained from log-mel spectrograms.

The main research question is:

> Does transfer learning with a pretrained PANNs CNN14 model provide better environmental sound classification performance than a compact CNN trained on log-mel spectrograms on the ESC-50 dataset?

The comparison will use accuracy, macro F1-score, confusion matrices, and inference time. The results will be interpreted as evidence on the ESC-50 benchmark rather than as a claim about all real-world environmental audio.

## 2. Project Type Decision

The selected project type is **Bring Your Own Method**. Two approaches will be implemented and compared under the same predefined data split:

1. A compact CNN trained on log-mel spectrograms.
2. A transfer-learning classifier trained on embeddings extracted from the pretrained PANNs CNN14 audio model using PyTorch.

The main experiment will use PANNs CNN14 as a frozen feature extractor.
Limited fine-tuning of the classifier head will only be attempted if the core
experiments and application are completed on schedule.

## 3. Project Summary and Intended Approach

Environmental sounds such as rain, barking dogs, sirens, footsteps, and keyboard typing have different but sometimes overlapping acoustic patterns.
This project will build a small classification system that predicts the sound category of a five-second recording.

For the CNN baseline, each audio clip will be decoded consistently and converted into a normalized log-mel spectrogram. A compact CNN with convolutional blocks, pooling, dropout, and a 50-class output layer will be
trained on these features.

For the transfer-learning approach, each clip will be converted to mono, resampled to 32 kHz, and passed through the pretrained PANNs CNN14 model in PyTorch. The resulting frame-level embeddings will be aggregated over time, for example by taking their mean, and a small classification head will be trained on the fixed-size representations.

Both approaches will use identical ESC-50 folds, class labels, validation procedures, and final test data. Random seeds and preprocessing settings will be recorded to support reproducibility. A web-application will allow a user to upload a WAV file, select a model, and inspect the top three predicted classes and their scores.

## 4. Dataset

The project will use the public [ESC-50 dataset](https://github.com/karolpiczak/ESC-50). It contains:

- 2,000 mono WAV recordings
- 50 environmental sound classes
- 40 clips per class
- approximately five seconds per clip
- predefined folds in the metadata for reproducible evaluation.

Examples of classes include dog, rain, sea waves, footsteps, keyboard typing, helicopter, siren, and glass breaking. The released audio is sampled at 44.1 kHz. The CNN pipeline will use a fixed spectrogram configuration, while the PANNs pipeline will resample the audio to the model's required 32 kHz mono format.

The dataset will be downloaded from its official repository. The predefined folds will be respected so that related clips do not leak between training and test data.

## 5. Scientific References

1. Piczak, K. J. (2015). *ESC: Dataset for Environmental Sound
	Classification*. Proceedings of the 23rd ACM International Conference on
	Multimedia, 1015-1018. https://doi.org/10.1145/2733373.2806390

	This paper introduces the ESC dataset and provides the benchmark context for the proposed experiments.

2. S. Hershey et al., "CNN architectures for large-scale audio classification," 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), New Orleans, LA, USA, 2017, pp. 131-135, doi: 10.1109/ICASSP.2017.7952132.
	https://doi.org/10.1109/ICASSP.2017.7952132

	This paper provides background for convolutional architectures and learned representations for audio classification.

3.  Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang and M. D. Plumbley, "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 2880-2894, 2020, https://doi.org/10.1109/TASLP.2020.3030497

	This paper describes the pretrained PANNs representations that will be used for the PyTorch transfer-learning approach.

## 6. Work-Breakdown Structure

The estimated workload is **approximately 60 hours**.

| Task | Activities | Estimate |
| --- | --- | ---: |
| Dataset collection and preparation | Download ESC-50, review its metadata, validate files, inspect the dataset and implement the fold split and preprocessing checks | 3 h |
| Literature review and experiment design | Read the scientific papers and review the PANNs implementation | 4 h |
| Network design and implementation | Implement log-mel features and the compact CNN in PyTorch, integrate PANNs CNN14 embedding extraction, and build the classifier head | 11 h |
| Training, validation, and fine-tuning | Train both approaches, select settings using validation data, optionally test limited classifier fine-tuning, and save the best checkpoints | 12 h |
| Evaluation and result analysis | Calculate accuracy and macro F1-score, create a confusion matrix, inspect errors, and measure training and inference time | 6 h |
| Application development | Build the web-based upload interface, validate WAV input, connect both models, display top-three predictions, and show a log-mel spectrogram | 7 h |
| Final report | Document the motivation, dataset, methods, experimental setup, results, limitations and references | 10 h |
| Presentation preparation | Select figures and results, prepare slides, and rehearse the presentation | 7 h |
| **Total** |  | **60 h** |
