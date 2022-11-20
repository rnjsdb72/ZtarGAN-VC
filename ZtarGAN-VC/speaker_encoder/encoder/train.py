from encoder.visualizations import Visualizations
import warnings
from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.model import SpeakerEncoder
from utils.profiler import Profiler
from pathlib import Path
import torch
from tqdm import tqdm

warnings.filterwarnings('ignore')

def sync(device: torch.device):
    # FIXME
    return 
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def train(cfgs):
    
    run_id = cfgs.run_id
    data_path = Path(cfgs.data_path)
    models_dir = Path(cfgs.models_dir)
    speaker_list_path = Path(cfgs.speaker_list_path)
    umap_every = cfgs.vis_every
    save_every = cfgs.save_every
    backup_every = cfgs.backup_every
    vis_every = cfgs.vis_every
    force_restart = cfgs.resume_from
    visdom_server = cfgs.visdom_server
    #      no_visdom: bool
    
    # Create a dataset and a dataloader
    dataset = SpeakerVerificationDataset(data_path, speaker_list_path)
    loader = SpeakerVerificationDataLoader(
        cfgs,
        dataset,
        cfgs.model.speakers_per_batch,
        cfgs.model.utterances_per_speaker,
        num_workers=0,
    )
    
    # Setup the device on which to run the forward pass and the loss. These can be different, 
    # because the forward pass is faster on the GPU whereas the loss is often (depending on your
    # hyperparameters) faster on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FIXME: currently, the gradient is None if loss_device is cuda
    loss_device = torch.device("cpu")
    
    # Create the model and the optimizer
    model = SpeakerEncoder(device, loss_device, cfgs)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfgs.model.learning_rate_init)
    init_step = 1
    
    # Configure file path for the model
    state_fpath = models_dir.joinpath(run_id + ".pt")
    backup_dir = models_dir.joinpath(run_id + "_backups")

    # Load any existing model
    if force_restart:
        if state_fpath.exists():
            print("Found existing model \"%s\", loading it and resuming training." % run_id)
            checkpoint = torch.load(state_fpath)
            model.load_state_dict(checkpoint, strict=False)
            init_step = checkpoint['step']
        else:
            print("No model \"%s\" found, starting training from scratch." % run_id)
    else:
        print("Starting the training from scratch.")
    model.train()
    
    # Initialize the visualization environment
    vis = Visualizations(run_id, vis_every, server=visdom_server)
    vis.log_dataset(dataset)
    vis.log_params(cfgs)
    device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    vis.log_implementation({"Device": device_name})
    
    # Training loop
    pbar = tqdm(loader)
    profiler = Profiler(summarize_every=10, disabled=False)
    for step, speaker_batch in enumerate(pbar, init_step):
        profiler.tick("Blocking, waiting for batch (threaded)")
        
        # Forward pass
        inputs = torch.from_numpy(speaker_batch.data).float().to(device)
        sync(device)
        profiler.tick("Data to %s" % device)
        embeds = model(inputs)
        sync(device)
        profiler.tick("Forward pass")
        embeds_loss = embeds.view((cfgs.model.speakers_per_batch, cfgs.model.utterances_per_speaker, -1)).to(loss_device)
        loss, eer = model.loss(embeds_loss)
        sync(loss_device)
        profiler.tick("Loss")

        # Backward pass
        model.zero_grad()
        loss.backward()
        profiler.tick("Backward pass")
        model.do_gradient_ops()
        optimizer.step()
        profiler.tick("Parameter update")
        
        # Update visualizations
        # learning_rate = optimizer.param_groups[0]["lr"]
        vis.update(loss.item(), eer, step, pbar)
        
        # Draw projections and save them to the backup folder
        if umap_every != 0 and step % umap_every == 0:
            backup_dir.mkdir(exist_ok=True)
            projection_fpath = backup_dir.joinpath("%s_umap_%06d.png" % (run_id, step))
            embeds = embeds.detach().cpu().numpy()
            vis.draw_projections(embeds, cfgs.model.utterances_per_speaker, step, projection_fpath)
            vis.save()

        # Overwrite the latest version of the model
        if save_every != 0 and step % save_every == 0:
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, state_fpath)
            
        # Make a backup
        if backup_every != 0 and step % backup_every == 0:
            backup_dir.mkdir(exist_ok=True)
            backup_fpath = backup_dir.joinpath("%s_bak_%06d.pt" % (run_id, step))
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, backup_fpath)
            
        profiler.tick("Extras (visualizations, saving)")
        