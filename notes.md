## Observations

### Critic Real, Critic Fake
Val Real and Fake loss are much lower might imply it already learn what is real and what is fake. what is the disadvantage of this?

without sgd dx goes to 0

never stop the training

batch norm shoudlnt be in the gen out and critic in -> after applying this still very bad and LeakyRelu on critic coldnt make it better

















Per action results

Start validation epoch
2d_2_3d Validation:             Loss: 0.0708    ReCon: 0.0042   KLD: 1.7719     critic_loss: 1.2320     gen_loss: 7.1504        D_x: 0.6343     D_G_z1: 0.0769  D_G_z2: 0.0769
2d_2_3d - * MPJPE * : 74.5449 
 per joint 
 tensor([  0.0000,  77.1236,  81.1871, 121.5773,  68.6131,  84.4415, 126.7168,
         50.5929,  43.1049,  54.2732,  73.1803,  49.2222,  80.5940, 109.1315,
         49.7943,  83.8382, 113.8718]) 

 [52.382102966308594, 70.56153869628906, 68.28710174560547, 64.94527435302734, 69.23880004882812, 57.96500778198242, 146.91574096679688, 93.15471649169922, 132.41453552246094, 68.19825744628906, 76.4754409790039, 71.19129943847656, 52.16413116455078, 69.61522674560547, 51.443939208984375]