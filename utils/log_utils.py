

def log_train_metrics(args, losses, lr, epoch, wandb, verbose=True):
    if verbose:
        print(f"Epoch {epoch}: Train loss: {losses[0]:.5f} | Train Hungarian: {losses[1]:.5f} | " +
               f"Train KL: {losses[2]:.5f} | " +
              (f"Train N: {losses[3]:.5f} | " if losses[3] != 0 else '') +
              f"Train lr: {lr:.1e}")

    if args.wandb:
        dic = {name: losses[i] for i, name in enumerate(["Train loss", "Hungarian loss", "KL loss", "Train N loss"])}
        dic['lr'] = lr
        wandb.log(dic)


def log_test_metrics(args, losses, epoch, wandb, verbose=True):
    if verbose:
        print(f"Epoch {epoch}: Test loss: {losses[0]:.5f} | Test Hungarian: {losses[1]:.5f} | " +
              (f" Test N: {losses[2]:.5f} | " if losses[2] != 0 else ''))

    if args.wandb:
        dic = {name: losses[i] for i, name in enumerate(["Test loss", "Test Hungarian", "Test N loss"])}
        wandb.log(dic)


def log_evaluation_metrics(args, losses, epoch, wandb, extrapolation):
    ext = 'Extrapolation ' if extrapolation else ""
    print(f"Epoch {epoch}: {ext} Bounding box loss: {losses[0]:.4f} | NN loss: {losses[1]:.4f} |" +
          f" Valency dist: {losses[2]:.4f} |" +
          f" N dist: {losses[3]:.3f} |  Diversity score: {losses[4]:.3f}")
    if args.wandb:
        wandb.log({f"{ext}Bounding box loss": losses[0],
                   f"{ext}NN loss": losses[1],
                   f"{ext}Valency dist": losses[2],
                   f"{ext}N dist": losses[3],
                   f"{ext}Diversity score": losses[4]})

