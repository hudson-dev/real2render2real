# Real2Render2Real
[project page](real2render2real.com) | [arxiv](https://arxiv.org/abs/2505.09601)

Justin Yu*, Letian Fu*, Huang Huang, Karim El-Refai, Rares Ambrus, Richard Cheng, Muhammad Zubair Irshad, Ken Goldberg

**Real2Render2Real: Scaling Robot Data Without Dynamics Simulation or Robot Hardware**

University of California, Berkeley | Toyota Research Institute
*Equal contribution

Real2Render2Real (R2R2R) is a scalable pipeline for generating robot training data without physics simulation or teleoperation. Using object scans and a single monocular video, R2R2R reconstructs 3D geometry and motion, then synthesizes thousands of diverse, physically plausible demonstrations for training generalist manipulation policies. Our experiments show that models trained on R2R2R data alone can achieve comparable performance to those trained on teleoperated demonstrations, while requiring a fraction of the time to generate.

## TODO
- Release sim-to-real pipeline (**September 15th, 2025**)
- Release the datasets to recreate results (**July 15th, 2025**)
- Release our policy training infrastructure (**June 15th, 2025**)

The code used to deploy high-frequency visuomotor policies on the YuMi IRB14000 robot is hosted on repo [uynitsuj/yumi_realtime](https://github.com/uynitsuj/yumi_realtime)
