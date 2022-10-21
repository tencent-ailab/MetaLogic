from copy import deepcopy
import re

# overall constant
task2prefix = {
    'tree': 'TREE: ',
    'triple': 'FORMULAE: ',
    'degree': 'DEGREE: ',
    'controller': 'CONTROL',
    'module': '',
    'module_clu': 'CONCLUSION: ',
    'module_reb': 'REBUTTAL: ',

}

task2flag = {
    'tree': '$tree$',
    'triple': '$formulae$',
    'degree': '$degree$', 
}


operator2token = {
    ### if use <extra_id_xx>, use skip_special_tokens=False when decode
    ### In preliminary experiments, we find that adding too many 
    ### special_tokens would make the model difficult to train and performs poorly
    # '[I-AND]': '<extra_id_99>',
    # '[I-OR]': '<extra_id_98>',
    # '[I-ENTAIL]': '<extra_id_97>',
    # '[NEG]': '<extra_id_96>',
    # '[BOX]': '<extra_id_95>',
    # '[DIAMOND]': '<extra_id_94>',

    '[I-AND]': '[and]',
    '[I-OR]': '[or]',
    '[I-ENTAIL]': '[entail]',
    '[NEG]': '[negative]',
    '[BOX]': '[necessary]',
    '[DIAMOND]': '[possible]',

    '[I-CONJUNCTION]': '[and]',
    '[I-DISJUNCTION]': '[or]',
    '[I-IMPLICATION]': '[entail]',
}
# token2operator = {v:k for k,v in operator2token.items()}
token2operator = {
    '[and]': '[I-CONJUNCTION]',
    '[or]': '[I-DISJUNCTION]',
    '[entail]': '[I-IMPLICATION]',
    '[negative]': '[NEG]', 
    '[necessary]': '[BOX]',
    '[possible]': '[DIAMOND]',
}

degree2token = {
    '0': 'impossible',
    '1': 'unnecessary',
    '2': 'contingent',
    '3': 'possible',
    '4': 'necessary',
}
token2degree = {v:k for k,v in degree2token.items()}


# process function
def preprocess_raw_data(datas):
    for data_item in datas:

        # merge context and option sentences
        sents = deepcopy(data_item['context_dict'])
        sents.update(data_item['option_dict'])
        
        for sent_id in sents.keys():
            inner_info = data_item['inner_formulae_and_sentences'][sent_id]
            
            fts = inner_info['formulae_triples']
            fts = [[operator_reduction(ft[0]),f"v{ft[1]}",ft[2].strip(),operator_reduction(ft[3]),f"v{ft[4]}"] for ft in fts]
            inner_info['formulae_triples'] = fts
            
            sents[sent_id]['inner_info'] = inner_info
        
        data_item['sent_dict'] = sents

        # context
        context_with_variable = ""
        for sent_id, sent_info in data_item['sent_dict'].items():
            context_with_variable += f"{sent_id}: {sent_info['inner_info']['inner_sent_w_variables']} "

        context_without_variable = ""
        for sent_id, sent_info in data_item['sent_dict'].items():
            context_without_variable += f"{sent_id}: {sent_info['sent']} "
        assert context_without_variable == data_item['context'] + ' ' + data_item['option'] + ' '

        data_item['context_with_variable'] = context_with_variable
        data_item['context_without_variable'] = context_without_variable

        # gold information
        proof_str = data_item['proof_without_and'].replace('[STEP_SPLITTER]', ';')
        proof = parse_proof(proof_str, spliter = ';', single_premise = True, flag = '')

        triples_dict = {}
        triples_str_dict = {}
        for sent_id, sent_info in data_item['sent_dict'].items():
            triples = sent_info['inner_info']['formulae_triples']
            triples_dict[sent_id] = triples

            triples_str = linearize_inner_triples(triples, flag = '')
            triples_str_map = transform_by_dict(triples_str, operator2token).strip()
            triples_str_dict[sent_id] = triples_str_map

        degree_dict = {}
        degree_str_dict = {}
        for sent_id, sent_info in data_item['sent_dict'].items():
            degree = sent_info['inner_info']['degree_label']
            degree_dict[sent_id] = degree

            degree_str = f"{degree}"
            degree_str_map = transform_by_dict(degree_str, degree2token).strip()
            degree_str_dict[sent_id] = degree_str_map

        gold_item = {
            'proof_str': proof_str,
            'triples_str_dict': triples_str_dict,
            'degree_str_dict': degree_str_dict,

            'proof': proof,
            'triples_dict': triples_dict,
            'degree_dict': degree_dict,
        }

        data_item['gold_item'] = gold_item


    return datas

def transform_by_dict(input_str, mapping_dict):
    for src, tgt in mapping_dict.items():
        input_str = input_str.replace(src, tgt)
    return input_str

def remove_given_tokens(input_str, remove_list=["</s>", "<pad>", "<unk>"]):
    for t in remove_list:
        input_str = input_str.replace(t, '')
    return input_str.strip()

def chunk(it, n):
    c = []
    for x in it:
        c.append(x)
        if len(c) == n:
            yield c
            c = []
    if len(c) > 0:
        yield c


# ----- proof -----
def parse_proof(proof, spliter = ';', single_premise = False, flag = '$tree$'):

    proof = proof.replace(flag, '').strip()

    steps = [step.strip() for step in proof.strip().split(spliter) if step.strip()]

    parsed_steps = []
    for step in steps:
        try:
            if '->' in step: step_type = '->'
            elif '=>' in step: step_type = '=>'
            else: continue

            pre, con = step.split(step_type)
            pre = sorted([p.strip() for p in pre.split('&')])
            con = con.strip()

            if single_premise == False:
                parsed_steps.append(
                    {
                        'step':step,
                        'pre':pre,
                        'con':con,
                        'type':step_type,
                    }
                )

            else:
                for p in pre:
                    parsed_steps.append(
                        {
                            'step':step,
                            'pre': [p],
                            'con':con,
                            'type':step_type,
                        }
                    )

        except Exception as e:
            # print(e)
            # print(proof, step)
            continue

    return parsed_steps

# ----- inner triple -----
def operator_reduction(operators):
    valid_operators = ["[NEG]", "[BOX]", "[DIAMOND]"]
    modal_final_ = [["[NEG]"], ["[BOX]"], ["[DIAMOND]"], ["[NEG]", "[BOX]"], ["[NEG]", "[DIAMOND]"]]

    try:
        operators = list(operators)

        if not all([opt in valid_operators for opt in operators]):
            return operators

        if len(operators) == 0:
            return operators
        else:
            if operators in modal_final_:
                return operators
            else:
                if operators[-3:] == ["[NEG]", "[BOX]", "[NEG]"]:
                    operators[-3:] = ["[DIAMOND]"]
                elif operators[-3:] == ["[NEG]", "[DIAMOND]", "[NEG]"]:
                    operators[-3:] = ["[BOX]"]
                elif operators[-3:] == ["[NEG]", "[NEG]", "[NEG]"]:
                    operators[-3:] = ["[NEG]"]
                elif operators[-3:] == ["[DIAMOND]", "[DIAMOND]", "[DIAMOND]"]:
                    operators[-3:] = ["[DIAMOND]"]
                elif operators[-3:] == ["[BOX]", "[BOX]", "[BOX]"]:
                    operators[-3:] = ["[BOX]"]

                if operators[-4:-1] == ["[NEG]", "[BOX]", "[NEG]"]:
                    operators[-4:-1] = ["[DIAMOND]"]
                elif operators[-4:-1] == ["[NEG]", "[DIAMOND]", "[NEG]"]:
                    operators[-4:-1] = ["[BOX]"]
                elif operators[-4:-1] == ["[NEG]", "[NEG]", "[NEG]"]:
                    operators[-4:-1] = ["[NEG]"]
                elif operators[-4:-1] == ["[DIAMOND]", "[DIAMOND]", "[DIAMOND]"]:
                    operators[-4:-1] = ["[DIAMOND]"]
                elif operators[-4:-1] == ["[BOX]", "[BOX]", "[BOX]"]:
                    operators[-4:-1] = ["[BOX]"]

                elif operators[-2:] == ["[BOX]", "[DIAMOND]"]:
                    operators[-2:] = ["[DIAMOND]"]
                elif operators[-2:] == ["[DIAMOND]", "[BOX]"]:
                    operators[-2:] = ["[BOX]"]
                elif operators[-2:] == ["[BOX]", "[BOX]"]:
                    operators[-2:] = ["[BOX]"]
                elif operators[-2:] == ["[DIAMOND]", "[DIAMOND]"]:
                    operators[-2:] = ["[DIAMOND]"]
                elif operators[-2:] == ["[NEG]", "[NEG]"]:
                    operators[-2:] = []

                elif operators[-3:-1] == ["[BOX]", "[DIAMOND]"]:
                    operators[-3:-1] = ["[DIAMOND]"]
                elif operators[-3:-1] == ["[DIAMOND]", "[BOX]"]:
                    operators[-3:-1] = ["[BOX]"]
                elif operators[3:-1] == ["[BOX]", "[BOX]"]:
                    operators[3:-1] = ["[BOX]"]
                elif operators[3:-1] == ["[DIAMOND]", "[DIAMOND]"]:
                    operators[3:-1] = ["[DIAMOND]"]
                elif operators[-3:-1] == ["[NEG]", "[NEG]"]:
                    operators[-3:-1] = []

                elif operators[-2:] == ["[BOX]", "[NEG]"]:
                    operators[-2:] = ["[NEG]", "[DIAMOND]"]
                elif operators[-2:] == ["[DIAMOND]", "[NEG]"]:
                    operators[-2:] = ["[NEG]", "[BOX]"]

                elif operators[-3:-1] == ["[BOX]", "[NEG]"]:
                    operators[-3:-1] = ["[NEG]", "[DIAMOND]"]
                elif operators[-3:-1] == ["[DIAMOND]", "[NEG]"]:
                    operators[-3:-1] = ["[NEG]", "[BOX]"]

                operators = operator_reduction(operators)

                return operators
    except Exception as e:
        print(e)
        print(operators)
        return []



def linearize_inner_triples(triples, flag='$formulae$'):
    s = flag + " "
    for t in triples:
        s += f"{' '.join(t[0])} {t[1]} {t[2]} {' '.join(t[3])} {t[4]}; "
    return s

def parse_inner_triples(triples_str, spliter = ';', flag = '$formulae$'):

    inner_relations = ['[I-AND]', '[I-OR]', '[I-ENTAIL]',
                        '[I-CONJUNCTION]', '[I-DISJUNCTION]', '[I-IMPLICATION]']

    triples_str = triples_str.replace(flag, '').strip()

    triples = [t.strip() for t in triples_str.split(spliter) if t.strip()]

    parsed_triples = []
    for triple in triples:
        try:
            triple_relation = None
            for ir in inner_relations:
                if ir in triple:
                    triple_relation = ir

            if triple_relation is None:
                continue

            pre, con = triple.split(triple_relation)

            pre = pre.split()
            pre_operators = operator_reduction([o.strip() for o in pre[:-1]])
            pre_var = pre[-1].strip()

            con = con.split()
            con_operators = operator_reduction([o.strip() for o in con[:-1]])
            con_var = con[-1].strip()

            parsed_triple = [pre_operators, pre_var, triple_relation, con_operators, con_var]
            parsed_triples.append(parsed_triple)
        except Exception as e:
            # print(e)
            # print(triples_str, triple)
            continue

    return parsed_triples

# ----- degree -----
def parse_degree(degree_str, flag = '$degree$'):

    degree_str = degree_str.replace(flag, '').strip()

    degree = -1
    try:
        degree = int(degree_str)
    except Exception as e:
        # print(e)
        # print(degree_str)
        degree = -1

    return degree


# ---- once -----
def parse_once(input_str):
    sent_spliter = "|"
    tree_slot = "tree"
    triple_slot = "formulae"
    degree_slot = "degree"


    # match slot
    slot_re = re.compile('(?i)'+re.escape("$SLOT$").replace("SLOT", "(\\w*?)"))
    slot_pos = []
    for m in slot_re.finditer(input_str):
        slot_pos.append((m.span(), m.group(1)))

    slot_dict = {}
    for idx, (pos, slot_name) in enumerate(slot_pos):
        if idx == len(slot_pos) - 1:
            value = input_str[pos[1]:]
        else:
            value = input_str[pos[1]:slot_pos[idx+1][0][0]]
        slot_dict[slot_name] = value

    # proof
    proof_str = slot_dict.get(tree_slot, '').strip()

    # triples
    all_triples_str = slot_dict.get(triple_slot, '').strip()
    triples_str_dict = {}
    for s in all_triples_str.split(sent_spliter):
        if len(s.split(':')) >= 2:
            sent_id, sent_triples_str = s.split(':')[-2:]
            triples_str_dict[sent_id.strip()] = sent_triples_str.strip()

    # degree
    all_degree_str = slot_dict.get(degree_slot, '').strip()
    degree_str_dict = {}
    for s in all_degree_str.split(sent_spliter):
        if len(s.split(':')) >= 2:
            sent_id, sent_degree_str = s.split(':')[-2:]
            degree_str_dict[sent_id.strip()] = sent_degree_str.strip()

    return {
        'str':input_str,
        'proof_str':proof_str,
        'triples_str_dict':triples_str_dict,
        'degree_str_dict':degree_str_dict,
        'slot_dict':slot_dict
    }

