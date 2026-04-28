from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import torch

from deepcase.preprocessing   import Preprocessor
from deepcase.context_builder import ContextBuilder
from deepcase.interpreter     import Interpreter


def load_text_to_df(path, machine_offset=0, label=None, nrows=None):
    """Read a DeepLog-format text file into a DataFrame.

    Each line = one machine's event sequence (space-separated integers).
    machine_offset avoids ID collisions when combining multiple files.
    """
    events, machines = [], []
    with open(path) as f:
        for machine, line in enumerate(f):
            if nrows is not None and machine >= nrows:
                break
            for event in map(int, line.split()):
                events.append(event)
                machines.append(machine + machine_offset)

    df = pd.DataFrame({
        'timestamp': np.arange(len(events)),
        'event'    : events,
        'machine'  : machines,
    })
    if label is not None:
        df['label'] = label
    return df


if __name__ == "__main__":

    ########################################################################
    #                             Loading data                             #
    ########################################################################

    preprocessor = Preprocessor(
        length  = 10,    # context window size
        timeout = 86400, # 1 day; has no real effect on index-based timestamps
    )

    # Load each split with unique machine IDs so they don't collide
    df_train = load_text_to_df(
        path           = 'data/hdfs/hdfs_train',
        machine_offset = 0,
        label          = 0,  # all normal
    )

    df_test_normal = load_text_to_df(
        path           = 'data/hdfs/hdfs_test_normal',
        machine_offset = int(df_train['machine'].max()) + 1,
        label          = 0,
    )

    df_test_abnormal = load_text_to_df(
        path           = 'data/hdfs/hdfs_test_abnormal',
        machine_offset = int(df_test_normal['machine'].max()) + 1,
        label          = 1,  # anomalous
    )

    n_train       = len(df_train)
    n_test_normal = len(df_test_normal)

    # Combine all three splits so the event→index mapping is consistent
    df_all = pd.concat(
        [df_train, df_test_normal, df_test_abnormal],
        ignore_index = True,
    )
    df_all['timestamp'] = np.arange(len(df_all))

    print("Processing sequences...")
    context_all, events_all, labels_all, mapping = preprocessor.sequence(
        df_all, verbose=True,
    )

    # Split back into train / test using original row counts
    context_train = context_all[:n_train]
    events_train  = events_all [:n_train]
    labels_train  = labels_all [:n_train]

    context_test  = context_all[n_train:]
    events_test   = events_all [n_train:]
    labels_test   = labels_all [n_train:]

    n_events = len(mapping)
    print(f"Unique event types (incl. NO_EVENT): {n_events}")

    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    context_train = context_train.to(device)
    events_train  = events_train .to(device)
    context_test  = context_test .to(device)
    events_test   = events_test  .to(device)

    ########################################################################
    #                         Training ContextBuilder                      #
    ########################################################################

    context_builder = ContextBuilder(
        input_size  = n_events,
        output_size = n_events,
        hidden_size = 128,
        max_length  = 10,
    ).to(device)

    context_builder.fit(
        X             = context_train,
        y             = events_train.reshape(-1, 1),
        epochs        = 10,
        batch_size    = 128,
        learning_rate = 0.01,
        verbose       = True,
    )

    ########################################################################
    #                          Training Interpreter                        #
    ########################################################################

    interpreter = Interpreter(
        context_builder = context_builder,
        features        = n_events,
        eps             = 0.1,
        min_samples     = 5,
        threshold       = 0.2,
    )

    # fit() internally calls cluster() → score_clusters() → score()
    interpreter.fit(
        X          = context_train,
        y          = events_train.reshape(-1, 1),
        scores     = labels_train.numpy(),  # all 0 (normal)
        iterations = 100,
        batch_size = 1024,
        strategy   = "max",
        NO_SCORE   = -1,
        verbose    = True,
    )

    ########################################################################
    #                              Evaluation                              #
    ########################################################################

    prediction = interpreter.predict(
        X          = context_test,
        y          = events_test.reshape(-1, 1),
        iterations = 100,
        batch_size = 1024,
        verbose    = True,
    )

    # prediction == 0  → matched a known-normal cluster → Normal
    # prediction <  0  → -1 (low confidence), -2 (unknown event),
    #                     -3 (outside all clusters) → treat as Anomaly
    y_pred = (prediction != 0).astype(int)
    y_true = labels_test.numpy()

    print("\n" + "=" * 60)
    print(classification_report(
        y_true       = y_true,
        y_pred       = y_pred,
        target_names = ['Normal', 'Anomaly'],
        digits       = 4,
    ))
