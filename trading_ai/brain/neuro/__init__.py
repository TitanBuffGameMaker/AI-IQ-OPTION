"""
Neuroscience-inspired modules for BrainCore.

การวิจัยสมองมนุษย์ → AI equivalent:
  CLS Memory    ← Hippocampus + Neocortex (Complementary Learning Systems)
  Dopamine      ← Reward Prediction Error (Wolfram Schultz, 1997)
  Fear System   ← Amygdala (risk aversion, emotional regulation)
  Sleep Cycle   ← Memory consolidation during slow-wave sleep
"""
from trading_ai.brain.neuro.cls_memory import CLSMemory
from trading_ai.brain.neuro.dopamine import DopamineSystem
from trading_ai.brain.neuro.fear_system import FearSystem
from trading_ai.brain.neuro.sleep_cycle import SleepCycle

__all__ = ["CLSMemory", "DopamineSystem", "FearSystem", "SleepCycle"]
