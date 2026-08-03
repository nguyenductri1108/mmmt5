"""The self-improvement loop.

Five mechanisms, all automatic, all bounded:

  memory.py      episodic memory — k-NN over your own closed trades; similar
                 past setups inform the AI gate and shrink size on bad evidence
  calibrator.py  measures whether the AI gate's rejections dodge losers;
                 auto-switches shadow/enforce and tunes min_confidence
  lessons.py     evidence-cited rules distilled weekly, injected into the gate
                 prompt; expire unless re-confirmed
  analyst.py     the weekly LLM review that writes lessons and proposes params
  optimizer.py   champion/challenger walk-forward parameter evolution with
                 out-of-sample validation, bootstrap testing, and auto-rollback

scheduler.py orchestrates them from inside the engine loop.

The invariant everything here obeys: learned adjustments can REJECT trades,
SHRINK size, or MOVE STRATEGY PARAMS WITHIN WHITELISTED BOUNDS. They can never
raise a risk cap, widen a stop, or grow a position.
"""

from core.learn.scheduler import LearningSystem

__all__ = ["LearningSystem"]
