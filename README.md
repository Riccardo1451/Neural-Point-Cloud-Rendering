# Neural Point Cloud Rendering

Progetto per il corso di Computer Graphics. Implementazione del paper "Neural Point Cloud Rendering via Multi-Plane Projection" sulle scene di 7-Scenes, con un confronto contro un renderer z-buffer classico quando si aggiunge rumore ai punti.

## Struttura

- `src/datasets`: lettura dei frame di 7-Scenes, point cloud da depth e fusione di più frame
- `src/voxelization`: voxelizzazione multi-piano, in NumPy (debug) e in PyTorch (training)
- `src/models`: feature dei punti, U-Net 3D e rete di rendering
- `src/losses`: perceptual loss con VGG-19
- `src/training`: training loop
- `src/baseline`: renderer z-buffer
- `src/eval`: metriche (PSNR, SSIM) e funzioni per l'inferenza
- `scripts`: script per debug, training, inferenza ed esperimenti
- `outputs*`: risultati salvati per le varie scene

## Come si usa

```bash
pip install -r requirements.txt

# point cloud della scena, da fare una volta sola
python scripts/precompute_scene_pointcloud.py --sequence_path data/raw/7scenes/chess/seq-01 --output_path data/processed/chess_seq01_pointcloud.pt

# training
python scripts/train.py --sequence_path data/raw/7scenes/chess/seq-01 --pointcloud_path data/processed/chess_seq01_pointcloud.pt --num_epochs 30

# confronto con lo z-buffer su alcuni frame
python scripts/infer.py --sequence_path data/raw/7scenes/chess/seq-01 --pointcloud_path data/processed/chess_seq01_pointcloud.pt --checkpoint_path checkpoints/checkpoint_latest.pt --frame_ids 0 50 200

# esperimento sul rumore
python scripts/noise_robustness_experiment.py --sequence_path data/raw/7scenes/chess/seq-01 --pointcloud_path data/processed/chess_seq01_pointcloud.pt --checkpoint_path checkpoints/checkpoint_latest.pt --frame_ids 0 50 100 150 200
```

Il dataset va scaricato a parte e messo in `data/raw/7scenes/`.
