from model import KWS_transformer
from preprocessing import collect_data
import tensorflow as tf
from argparse import ArgumentParser
import tensorflow_addons as tfa



def main(args):
    """
    Main function to train the model
    :param args:
    :return: saved model directory
    """
    data_dir = args.data_dir
    train_ds, test_ds, val_ds, commands = collect_data(batch_size=args.batch_size, data_dir=data_dir)
    print("train_ds, test_ds, val_ds uploaded")
    model = KWS_transformer(
        image_size=(40, 98),
        patch_size=(20, 2),
        num_layers=args.num_layers,
        num_classes=len(commands),
        d_model=args.d_model,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        dropout=0
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        optimizer=tfa.optimizers.AdamW(
            learning_rate=args.lr, weight_decay=args.weight_decay
        ),
        metrics=["accuracy"],
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.logdir, histogram_freq=1)
    with tf.device('/device:GPU:0'):
        model.fit(
            train_ds,
            epochs=args.epochs,
            validation_data=val_ds,
            callbacks=[tensorboard_callback]
        )
    model.save(args.save_dir)
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test loss: {loss}, Test Accuracy:{accuracy}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--num_layers", default=12, type=int)
    parser.add_argument("--d_model", default=192, type=int)
    parser.add_argument("--num_heads", default=3, type=int)
    parser.add_argument("--mlp_dim", default=768, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--save_dir", default="KWS_transformer")
    args = parser.parse_args()

    main(args)