"""
Unit tests for the Child-to-Genius Graphic Designer Curriculum Learning Engine.
"""

import unittest
from PIL import Image
from pipeline.designer_curriculum import GraphicDesignerCurriculum


class TestGraphicDesignerCurriculum(unittest.TestCase):

    def test_level1_primitive(self):
        img, desc, cot = GraphicDesignerCurriculum.generate_level1_primitive()
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (256, 256))
        self.assertIn("graphic design", desc.lower())
        self.assertTrue(cot.startswith("<think>"))
        self.assertIn("</think>", cot)
        self.assertIn("Level 1", cot)

    def test_level2_flashcard_icon(self):
        img, desc, cot = GraphicDesignerCurriculum.generate_level2_flashcard_icon()
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (256, 256))
        self.assertTrue(cot.startswith("<think>"))
        self.assertIn("</think>", cot)
        self.assertIn("Level 2", cot)
        self.assertIn("Aspect:", cot)

    def test_level3_layout_materials(self):
        img, desc, cot = GraphicDesignerCurriculum.generate_level3_layout_materials()
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (256, 256))
        self.assertTrue(cot.startswith("<think>"))
        self.assertIn("</think>", cot)
        self.assertIn("Level 3", cot)

    def test_level4_genius_masterpiece(self):
        img, desc, cot = GraphicDesignerCurriculum.generate_level4_genius_masterpiece()
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (256, 256))
        self.assertTrue(cot.startswith("<think>"))
        self.assertIn("</think>", cot)
        self.assertIn("Level 4", cot)

    def test_unified_dispatcher(self):
        for lvl in [1, 2, 3, 4, 0]:
            img, desc, cot = GraphicDesignerCurriculum.generate_curriculum_sample(lvl)
            self.assertEqual(img.size, (256, 256))
            self.assertGreater(len(desc), 10)
            self.assertIn("<think>", cot)
            self.assertIn("</think>", cot)


if __name__ == "__main__":
    unittest.main()
