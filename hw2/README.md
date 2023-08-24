# CS725: Homework 2: Classification using neural networks

![num_epochs=250 learning_rate=0 0005 20230822_11-33-01 training anim](https://github.com/ashutoshbsathe/cs725-hw/assets/22210756/d562ebfe-9016-48d9-b9f8-3d67384fd27d)

## TAs in-charge
* Ashutosh Sathe [@ashutoshbsathe](https://github.com/ashutoshbsathe)
* Krishnakant Bhatt [@KKBhatt17](https://github.com/KKBhatt17)

## Setting up
Make sure you have a python environment setup with latest versions of dependencies as listed in [`requirements.txt`](requirements.txt). If the latest versions don't work for you or you are unsure about the versions, contact the TAs.

## Instructions
* Implement both the models (on "simple" and "digits" dataset) in [`model.py`](model.py). You should ideally modify only the functions in this file. If you think you need to modify anything else, contact the TAs.
* To train your models, use the following commands:

For "simple" dataset:
```
python train.py --dataset simple --model simple --num_epochs <num_epochs> --learning_rate <learning_rate> --seed <seed>
```

For "digits" dataset:
```
python train.py --dataset digits --model digits --num_epochs <num_epochs> --learning_rate <learning_rate> --seed <seed>
```

The default values for each of these parameters are available in [`args.py`](args.py). The `seed` argument is useful if your model uses random initialization (default PyTorch behavior). With a fixed seed, your experiment will be reproducible. You can fix a seed throughout your experiments for best reproducibility. It is possible that a different `seed` will give different result but the differences will mostly be minor so you should focus on choosing better hyperparameters and making a better model.

## TensorBoard
This assignment uses PyTorch Lightning which allows us to use a far more sophisticated logger called [`TensorBoard`](https://www.tensorflow.org/tensorboard) developed originally for the TensorFlow framework. To view your training logs, you can go into `checkpoints/` directory and run `tensorboard --logdir ./` to start TensorBoard. You can now view the (live) training progress of all your models by going to `http://localhost:6006` in your browser! You can also download the data from here in a CSV for making plots in your report.

## Kaggle Submission
You can compete on the Kaggle competition by creating a submission using `python make_kaggle_submission.py <path-to-best-digits-ckpt>`. This will create a file called [`kaggle_upload.csv`](kaggle_upload.csv) which you can upload on Kaggle to see results.

## Moodle Submission
Once you are done with both the tasks, copy weights corresponding to your best models for each dataset into [`submission/`](submission/) directory. Also copy the completed version (implementing both models) of [`model.py`](model.py) and your observation report (with filename `report.pdf`) into the same directory. Overall, make sure your [`submission`](submission/) folder looks as below. This is crucial since the assignment will be autograded:
```
submission/
    model.py
    best_simple.ckpt
    best_digits.ckpt
    report.pdf
```
You can get a hint of the accuracy and loss values that autograder will use for grading your submission by running `python evaluate_submission.py` in this directory itself. The observation report should also contain roll numbers of both students in the team.

Once you are satisfied with the submission, use `tar -cvzf <roll1>_<roll2>.tar.gz submission/` to create the final submission. Only the student with lower roll number in the group needs to upload this `.tar.gz` on Moodle.

## Visualizing (Optional)

Once you have implemented the `LitSimpleClassifier` on the "simple" dataset, you can use the file [`train_with_visualization.py`](train_with_visualization.py) instead of standard [`train.py`](train.py) to create the GIF demonstrating the evolution of decision boundary of the classifier. You can find such a GIF at the top of this README. Use `python train_with_visualization.py -h` to see the list of options available to you for customizing the GIF output and training parameters. Do note that this is an optional task meant only for improving your understanding and may require significantly more computational power than just training the model.
