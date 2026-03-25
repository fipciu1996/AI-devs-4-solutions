import unittest

from reactor.solve_reactor import BlockState, ReactorState, plan_commands


class SolveReactorTests(unittest.TestCase):
    def test_block_advance_completes_full_cycle(self) -> None:
        start = BlockState(col=2, top_row=1, bottom_row=2, direction="down")
        current = start

        for _ in range(6):
            current = current.advance()

        self.assertEqual(current, start)

    def test_step_rejects_move_into_descending_block(self) -> None:
        state = ReactorState(
            player_col=1,
            blocks=(
                BlockState(col=2, top_row=3, bottom_row=4, direction="down"),
                BlockState(col=3, top_row=4, bottom_row=5, direction="up"),
                BlockState(col=4, top_row=3, bottom_row=4, direction="up"),
                BlockState(col=5, top_row=2, bottom_row=3, direction="down"),
                BlockState(col=6, top_row=2, bottom_row=3, direction="down"),
            ),
        )

        self.assertIsNone(state.apply("right"))

    def test_step_allows_entering_column_when_block_moves_up(self) -> None:
        state = ReactorState(
            player_col=1,
            blocks=(
                BlockState(col=2, top_row=4, bottom_row=5, direction="up"),
                BlockState(col=3, top_row=2, bottom_row=3, direction="down"),
                BlockState(col=4, top_row=3, bottom_row=4, direction="up"),
                BlockState(col=5, top_row=4, bottom_row=5, direction="up"),
                BlockState(col=6, top_row=4, bottom_row=5, direction="up"),
            ),
        )

        next_state = state.apply("right")

        self.assertIsNotNone(next_state)
        assert next_state is not None
        self.assertEqual(next_state.player_col, 2)
        self.assertEqual(next_state.blocks[0], BlockState(2, 3, 4, "up"))

    def test_step_allows_escaping_before_block_descends(self) -> None:
        state = ReactorState(
            player_col=2,
            blocks=(
                BlockState(col=2, top_row=3, bottom_row=4, direction="down"),
                BlockState(col=3, top_row=4, bottom_row=5, direction="up"),
                BlockState(col=4, top_row=2, bottom_row=3, direction="down"),
                BlockState(col=5, top_row=4, bottom_row=5, direction="up"),
                BlockState(col=6, top_row=4, bottom_row=5, direction="up"),
            ),
        )

        next_state = state.apply("left")

        self.assertIsNotNone(next_state)
        assert next_state is not None
        self.assertEqual(next_state.player_col, 1)
        self.assertTrue(next_state.blocks[0].occupies_bottom_lane())

    def test_plan_commands_reaches_goal_from_sample_state(self) -> None:
        state = ReactorState(
            player_col=1,
            blocks=(
                BlockState(col=2, top_row=1, bottom_row=2, direction="down"),
                BlockState(col=3, top_row=2, bottom_row=3, direction="down"),
                BlockState(col=4, top_row=4, bottom_row=5, direction="up"),
                BlockState(col=5, top_row=3, bottom_row=4, direction="up"),
                BlockState(col=6, top_row=4, bottom_row=5, direction="up"),
            ),
        )

        commands = plan_commands(state)

        self.assertIsNotNone(commands)
        assert commands is not None
        current = state
        for command in commands:
            next_state = current.apply(command)
            self.assertIsNotNone(next_state)
            assert next_state is not None
            current = next_state

        self.assertTrue(current.is_goal())


if __name__ == "__main__":
    unittest.main()
