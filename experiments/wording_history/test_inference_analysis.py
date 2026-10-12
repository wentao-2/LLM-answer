"""Analytic fixtures verify future cohort and common-denominator analyses."""
import copy
import unittest
import analyze_inference as a


def record(name, probability, item):
    return dict(model='fixture',status='ok',probability=probability,metadata={},item=dict(item,id=name))


class InferenceAnalysisTests(unittest.TestCase):
    def paraphrase_fixture(self):
        rows=[]
        world=dict(world_id='w',family_id='f',template_id='likelihood_v3',signal=1,answer=1,
            protocol='independent',visibility='signal_visible',reminder='legacy',output_instruction='restrictive',
            prior=.5,reliability=.75,oracle=.9)
        for version,values in (('original',(.8,.7,.9)),('authorship',(.9,.8,.9)),('dependence',(.9,.9,.9))):
            for author,p in zip(('self','other','neutral'),values):
                rows.append(record(version+'-'+author,p,dict(world,paraphrase_variant=version,attribution=author)))
        return rows

    def test_version_changes_and_errors(self):
        result=a.paraphrase_results(self.paraphrase_fixture())['strata'][0]
        self.assertEqual(result['complete_nine_sets'],1)
        self.assertAlmostEqual(result['versions']['original']['gap']['point']['estimate'],.1)
        self.assertAlmostEqual(result['versions']['original']['self_error']['point']['estimate'],-.1)
        self.assertAlmostEqual(result['versions']['original']['other_error']['point']['estimate'],-.2)
        self.assertAlmostEqual(result['paired_version_changes']['authorship']['point']['estimate'],0)
        self.assertAlmostEqual(result['paired_version_changes']['dependence']['point']['estimate'],-.1)

    def test_neutral_missing_disables_common_point_not_identified_gap(self):
        rows=self.paraphrase_fixture()
        rows[2].update(status='invalid',probability=None)
        result=a.paraphrase_results(rows)['strata'][0]
        gap=result['versions']['original']['gap']
        self.assertEqual(result['complete_nine_sets'],0)
        self.assertIsNone(gap['point']['estimate'])
        self.assertAlmostEqual(gap['bounds']['lower'],.1)
        self.assertAlmostEqual(gap['bounds']['upper'],.1)

    def test_generated_wrong_commitment_can_be_corrected(self):
        parent=record('p',.2,dict(family_id='f',parent_condition='numeric_off',oracle=.75))
        parent['metadata']['final_output']='FINAL_JSON: {"p_h1":0.2}'
        children=[]
        for evidence,b,q,protocol,visibility in (('none',None,.75,'baseline','baseline'),
            ('copy',1,.75,'positive','signal_visible'),('independent',0,.5,'independent','signal_visible'),
            ('independent',1,.9,'independent','signal_visible')):
            for arm in ('native_self','replay_self','replay_other','replay_neutral'):
                item=dict(parent_item_id='p',parent_model='fixture',history_arm=arm,evidence_kind=evidence,
                    next_signal=b,initial_answer=0,family_id='f',template_id='likelihood_v3',signal=1,
                    answer=b,oracle=q,protocol=protocol,visibility=visibility)
                children.append(record(evidence+str(b)+arm,q,item))
        original=copy.deepcopy(children)
        result=a.history_results([parent],children)
        self.assertEqual(children,original)
        self.assertEqual(result['attained_cohorts']['wrong_choice'],1)
        self.assertEqual(result['attained_cohorts']['overconfident'],1)
        cohort=next(r for r in result['cohorts'] if r['cohort']=='wrong_choice')
        for value in cohort['individual_update_errors']:
            self.assertAlmostEqual(value['absolute']['estimate'],0)
        self.assertAlmostEqual(cohort['no_new_evidence_change_from_initial']['native_self']['estimate'],-.55)
        empty=next(r for r in result['cohorts'] if r['cohort']=='correct_choice')
        self.assertEqual(empty['outcome'],'empty_cohort_not_tested')


if __name__=='__main__':
    unittest.main()
