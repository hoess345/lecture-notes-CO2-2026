import keras
from datetime import datetime

def train_model(model, X_train, y_train, model_name, epochs=50, early_stopping=True, patience=50, verbose = 1):
    
    log_dir = f"logs/{model_name}"
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    callbacks = [tensorboard_callback]
    
    if early_stopping:
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=verbose
    )
    
    return model, history



def eval_model(model, X_test, y_test, model_name=None):
    """Evaluate model on test data and print summary."""
    from sklearn.metrics import r2_score
    
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0)
    r2 = r2_score(y_test, y_pred)
    
    if model_name:
        print(f"\n{'='*50}")
        print(f"  {model_name} - Test Results")
        print(f"{'='*50}")
    else:
        print(f"\n{'='*50}")
        print(f"  Test Results")
        print(f"{'='*50}")
    
    print(f"  Test Loss (MSE): {test_loss:,.2f}")
    print(f"  Test MAE:        {test_mae:.2f}")
    print(f"  R² Score:        {r2:.4f} ({r2*100:.2f}% variance explained)")
    print(f"{'='*50}\n")
    
    return test_loss, test_mae, r2


def eval_model_classification(model, X_test, y_test, model_name=None, threshold=0.5):
    """Evaluate a classification model and print a confusion matrix."""
    import numpy as np
    from sklearn.metrics import (
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
    )

    results = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
    y_score = model.predict(X_test, verbose=0)

    if y_score.ndim == 2 and y_score.shape[1] > 1:
        y_pred = np.argmax(y_score, axis=1)
    else:
        y_pred = (y_score.reshape(-1) >= threshold).astype(int)

    y_true = np.asarray(y_test).reshape(-1).astype(int)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)

    if model_name:
        print(f"\n{'='*50}")
        print(f"  {model_name} - Test Results")
        print(f"{'='*50}")
    else:
        print(f"\n{'='*50}")
        print("  Test Results")
        print(f"{'='*50}")

    if "loss" in results:
        print(f"  Test Loss:         {results['loss']:.4f}")
    print(f"  Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"  F1 Score:          {f1:.4f}")
    print(f"  MCC:               {mcc:.4f}")
    print("\n  Confusion Matrix")
    print("  Rows: true labels, columns: predicted labels")
    print(matrix)
    print(f"{'='*50}\n")

    return results, balanced_accuracy, f1, mcc, matrix



def start_tensorboard(logs_dir='logs', port=6006):
    """
    Display instructions for starting TensorBoard.
    
    Args:
        logs_dir: Directory containing TensorBoard logs (default: 'logs')
        port: Port to run TensorBoard on (default: 6006)
    """
    import os
    import socket
    
    # Get absolute path to logs
    logs_path = os.path.abspath(logs_dir)
    
    if not os.path.exists(logs_path):
        print(f"⚠️  Warning: Logs directory not found at {logs_path}")
        return
    
    # Get hostname and construct URL
    hostname = socket.gethostname()
    url = f"https://{hostname}-{port}.swedencentral.instances.azureml.ms"
    
    print("=" * 70)
    print("📊 TensorBoard Setup Instructions")
    print("=" * 70)
    print("\n1. Open a terminal in Azure ML Studio")
    print("\n2. Run this command:")
    print(f"\n   tensorboard --logdir {logs_path} --host 0.0.0.0 --port {port}\n")
    print("3. Open this URL in your browser:")
    print(f"\n   {url}\n")
    print("=" * 70)
    
    return url