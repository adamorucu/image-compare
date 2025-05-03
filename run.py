from train import train
from eval import evaluate
from config import TrainingConfig

def run():
    config = TrainingConfig(
        model_type="cnn",
        model_config="shallow",
        dataset="cifar10",
        batch_size=128,
        learning_rate=0.001,
        epochs=5,
        save_every=5,
        use_wandb=False,
        device="cpu",
        early_stopping=10
    )
    loc = train(config)
    # loc = 'checkpoints/cnn_shallow_epoch5.pt'
    evaluate(config, loc)

if __name__ == "__main__":
    run()