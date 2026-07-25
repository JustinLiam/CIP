import numpy as np

from scripts.epi_abm.evaluate_mini_abm_oracle import action_grid, training_return_loss


def test_action_grid_is_stable_cartesian_product():
    grid = action_grid([0.0, 0.5, 1.0])
    assert grid.shape == (9, 2)
    np.testing.assert_array_equal(grid[0], [0.0, 0.0])
    np.testing.assert_array_equal(grid[4], [0.5, 0.5])
    np.testing.assert_array_equal(grid[8], [1.0, 1.0])


def test_training_return_loss_matches_discounted_clipped_negative_outcome_reward():
    loss = training_return_loss(
        np.asarray([0.0, 2.0, 10.0]),
        1.0,
        tau=3,
        discount=0.5,
        reward_clip=3.0,
    )
    assert loss == 1.0 + 0.5 * 1.0 + 0.25 * 3.0


def test_training_return_loss_is_not_terminal_only():
    early = training_return_loss([1.0, 1.0, 1.0], 1.0, tau=3, discount=0.99, reward_clip=3.0)
    late = training_return_loss([0.0, 0.0, 1.0], 1.0, tau=3, discount=0.99, reward_clip=3.0)
    assert early < late
