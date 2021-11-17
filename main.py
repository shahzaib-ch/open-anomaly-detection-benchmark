from core.core import do_benchmarking

if __name__ == '__main__':
    # pass desired training dataset size e.g. 0.3 means 30 percent
    do_benchmarking(training_dataset_size=0.3, do_not_update_existing_result=True)
