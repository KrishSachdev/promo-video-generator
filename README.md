Automated Promo Video Generator using CNN

This project automates the creation of short promotional videos from long-form video content using computer vision and audio analysis techniques. It leverages a pretrained convolutional neural network to identify visually important segments and combines them with audio-based scoring to generate high-impact video summaries.

The system uses a pretrained EfficientNetB0 CNN for visual feature extraction, along with OpenCV and NumPy for frame sampling and preprocessing. Audio features are extracted and integrated with visual scores to detect peaks representing engaging moments. Selected segments are then assembled into a concise promotional video using MoviePy.

This project demonstrates practical application of transfer learning, basic computer vision concepts, and end-to-end pipeline development in Python, with a focus on real-world video processing tasks.

https://foml-app-xt6iqeuawsldapwfdipsca.streamlit.app/
