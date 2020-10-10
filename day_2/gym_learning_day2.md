## 强化学习:Q-learning 原理

> Q-learning note
  https://www.zhihu.com/column/qlearning

> Q-learning format basic
  https://segmentfault.com/a/1190000007813298

> deep reinforcement learning course
  https://simoninithomas.github.io/deep-rl-course/   


## Note 
1. 當中Gamma 值要在0,1 之間, 當gamma越接近0 , 對立即的獎勵更有效.  當gamma越接近1 , 整個系統會更考慮將來的獎勵. 
`Q（狀態，動作）= R（狀態，動作）+ Gamma *最大[ Q（下一個狀態，所有動作）] `