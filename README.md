# ezpz-bci
The official repository for the "ezpz-bci" project - BrainHack 2025 Brisbane. 


This project develops a customization interface that translates EEG signals into computer inputs. Users can train the system to recognise distinct brainwave patterns (mental/motor imagery), each of which corresponds to a configurable metric/class. These metrics are then dynamically mapped to specific computer keys (or mouse input), enabling EEG-based interaction with games, applications, or custom software environments.

Behind the scenes is the EEGNet classifier. Find out more about the classifier [here](https://github.com/Dooganar/bioe6100-MI-EEG-classification). The aim is that this application will train the model locally on the users machine, and use it to predict the class of the EEG data, and then map it to the a computer input (keyboard or mouse). The user can then theoretically use their EEG to control any program on their computer, and have full control over the interface. We currently support the Neurosity Crown and the OpenBCI Cyton devices, but this can be easily expanded to more. This project is committed to free and open source practices for it's entire lifecycle.
