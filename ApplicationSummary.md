## EZPZ-BCI: A Customizable Interface for EEG-Based Computer Input

When utilizing EEG signals to classify user intent three common issues arise:

1. Closed EEG hardware and limited access to live data. Many consumer-grade EEG devices provide little to no access to raw real-time data, often relying on closed interfaces with minimal customizability or API access.
2. Lack of accessible classification tools. Even with open-source hardware or API access, translating raw EEG signals into actionable insights requires significant technical expertise in signal processing and machine learning.
3. Limited interoperability. Successfully classified EEG data rarely integrates seamlessly with everyday tools, such as standard operating systems, applications or medical devices/interfaces.

EZPZ-BCI was developed at the OHBM Hackathon to address the latter two challenges. It is a lightweight, user-friendly EEG classification and control interface/toolkit that enables users to train the system to recognise distinct brainwave patterns (mental/motor imagery) measured via EEG signals and map these distinct mental states to customizable computer inputs (e.g., keyboard commands), creating new opportunities for control  and interaction—without requiring deep technical knowledge.

Our prototype allows users to train EEG-based classification models using the motor imagery tasks left-hand clench, right-hand clench, and rest. Where rest acts as an idle state. Once trained on the users EEG data, these thoughts can be mapped to any keyboard input, enabling EEG-based control of games, applications, or assistive tools. The system features a simple front-end interface and is designed for plug-and-play use with consumer-grade EEG devices that provide access to raw live EEG data.

In the limited timeframe of the Hackathon, we successfully tested EZPZ-BCI with two devices: the OpenBCI Ganglion and the Neurosity Crown. Users can connect either device, train their model using the provided GUI, and map each classified brain state to a corresponding keyboard key. For example, a left-hand clench thought could trigger a left arrow key press. With this system, users can navigate interfaces or control games using only mental commands.

The classifier is based on a modified version of EEGNet [1], a compact convolutional neural network (CNN) architecture specifically designed for EEG-based BCI tasks. EEGNet leverages core neurophysiological principles—temporal filtering, spatial filtering, and frequency-band separation—while maintaining a small number of trainable parameters, making it well-suited for real-time, low-power applications [2].

The next steps for the project include:
1. Expanding training capabilities to support arbitrary user-defined thoughts beyond motor imagery (Left Hand Clench, Right Hand Clench and Rest).
2. Packaging the toolkit into a standalone executable application to improve accessibility for non-technical users.
3. Enhancing the front-end to support intuitive model training, testing, and key mapping workflows.

We envision EZPZ-BCI as a tool for a wide range of use cases, including hobbyist and research experimentation, assistive technology for individuals with mobility impairments, low-cost cognitive training and rehabilitation, and exploratory research on user agency in conditions such as paralysis.

In the coming months we intend on finalising the above action items and hope to have a fully complete executable. 

### Contibutors
* Reuben Richardson
* Isabel Barton
* Mac Rogers
* Benjamin Pettit
* Ayman Diallo

### Resources
UQ Neurotech discord server: https://discord.gg/j8zRpvUx2Z

### References
[1] Dooganar, bioe6100-MI-EEG-classification, GitHub repository, July 2025, [Online]. Available:
https://github.com/Dooganar/bioe6100-MI-EEG-classification.git

[2] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, “EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces,” May 2018, arXiv:1611.08024 [cs]. [Online]. Available:
http://arxiv.org/abs/1611.08024

