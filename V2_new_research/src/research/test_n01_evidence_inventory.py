import unittest
from n01_evidence_inventory import inventory

class InventoryTests(unittest.TestCase):
    def test_negation_uncertainty_overlap(self):
        x = inventory('Cannot exclude an acute fracture line.')['Fracture']
        self.assertEqual((x['mention'], x['negation'], x['uncertainty']), (1, 1, 1))
        x = inventory('Fracture is not excluded.')['Fracture']
        self.assertEqual((x['negation'], x['uncertainty']), (1, 1))

    def test_explicit_negative_is_only_a_cue(self):
        x = inventory('No joint effusion.')['Effusion']
        self.assertEqual(x['negation'], 1)
        self.assertNotIn('negative', x)
        self.assertNotIn('mask', x)

    def test_multilingual_and_severity(self):
        x = inventory('Sin derrame articular.')['Effusion']
        self.assertEqual((x['mention'], x['negation']), (1, 1))
        x = inventory('Moderate joint effusion, possible synovitis.')['Effusion']
        self.assertEqual((x['severity'], x['uncertainty']), (1, 1))

    def test_distance_silence_and_dimension(self):
        x = inventory('No fracture.' + ' filler'*80 + 'Possible joint effusion.')['Fracture']
        self.assertEqual((x['negation'], x['uncertainty']), (1, 0))
        self.assertEqual(inventory('Normal muscle.')['Synovitis']['mention'], 0)
        self.assertEqual(inventory('Medial compartment cartilage loss measuring 1.2 cm.')['Medial OA']['dimension'], 1)

if __name__ == '__main__':
    unittest.main()
