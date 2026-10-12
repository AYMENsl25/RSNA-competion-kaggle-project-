import unittest
from n01_assertions import extract

class AssertionTests(unittest.TestCase):
    def test_silence_and_language(self):
        self.assertEqual(extract('Normal muscle signal.', 'en')['Fracture'], 'not-addressed')
        self.assertTrue(all(v == 'not-addressed' for v in extract('ACL tear.', 'es').values()))

    def test_clause_scope(self):
        result = extract('No effusion, synovitis is present. No fracture but traumatic bone contusion without fracture is present.', 'en')
        self.assertEqual(result['Effusion'], 'negative')
        self.assertEqual(result['Synovitis'], 'positive')
        self.assertEqual(result['Fracture'], 'negative')
        self.assertEqual(result['Contusion'], 'positive')

    def test_uncertainty_and_history(self):
        for report in ('Cannot exclude fracture.', 'Possible ACL tear.', 'Prior ACL tear.',
                       'ACL reconstruction.', 'ACL tear?'):
            target = 'Fracture' if 'fracture' in report else 'ACL'
            self.assertEqual(extract(report, 'en')[target], 'uncertain')
        result = extract('History: ACL tear\nFindings: ACL intact.', 'en')
        self.assertEqual(result['ACL'], 'negative')

    def test_anatomy_and_conflict(self):
        self.assertEqual(extract('Medial meniscus tear. Lateral meniscus normal.', 'en')['Medial Meniscus'], 'positive')
        self.assertEqual(extract('ACL intact and MCL torn.', 'en')['ACL'], 'uncertain')
        self.assertEqual(extract('Effusion present. No effusion.', 'en')['Effusion'], 'uncertain')
        self.assertEqual(extract('ACL degeneration.', 'en')['ACL'], 'uncertain')
        self.assertEqual(extract('ACL is not intact.', 'en')['ACL'], 'uncertain')

    def test_oa_mask(self):
        self.assertEqual(extract('Severe medial compartment osteoarthritis.', 'en')['Medial OA'], 'uncertain')
        self.assertEqual(extract('Normal patellofemoral cartilage.', 'en')['PF OA'], 'negative')
        self.assertEqual(extract('Medial compartment full thickness cartilage loss measuring 1.2 cm.', 'en')['Medial OA'], 'positive')
        self.assertEqual(extract('Medial compartment full thickness cartilage loss measuring 5 mm.', 'en')['Medial OA'], 'uncertain')

    def test_thresholds(self):
        for target, report in [('ACL', 'ACL tear.'), ('MCL', 'Complete MCL tear.'),
                               ('Effusion', 'Small joint effusion.'), ("Baker's", 'Baker cyst.'),
                               ('Fracture', 'Acute fracture.'), ('Contusion', 'Bone contusion.')]:
            self.assertEqual(extract(report, 'en')[target], 'uncertain')
        for target, report in [('ACL', 'Complete ACL tear.'), ('MCL', 'Acute complete MCL tear.'),
                               ('Effusion', 'Moderate joint effusion.'), ("Baker's", 'Large Baker cyst.'),
                               ('Fracture', 'Acute fracture line.')]:
            self.assertEqual(extract(report, 'en')[target], 'positive')

if __name__ == '__main__':
    unittest.main()
