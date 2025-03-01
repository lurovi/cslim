import random
import torch
from cslim.algorithms.GSGP.representations.tree import Tree
from cslim.algorithms.SLIM_GSGP.representations.individual import Individual


def generate_mask(n, k):
    my_mask = [0] * n
    positions = set()
    while len(positions) < k-1:
        positions.add(random.randint(1, n-1))
    for pos in positions: 
        my_mask[pos] = 1
    #the first block needs to always be saved(root block)
    my_mask[0] = 1
    return my_mask

def std_xo_alpha_delta(operator='sum'):

    def stdxo_a_delta(p1, p2, alpha, testing):

        if isinstance(p1, Tree):
            if testing:
                p1 = p1.test_semantics
                p2 = p2.test_semantics
            else:
                p1 = p1.train_semantics
                p2 = p2.train_semantics



        return torch.add(torch.mul(p1, alpha), torch.mul(torch.sub(1, alpha), p2)) if operator == 'sum' else \
                torch.mul(torch.pow(p1, alpha), torch.pow(p2, torch.sub(1,  alpha)))



    return stdxo_a_delta

def std_xo_alpha_ot_delta(which, operator='sum'):

    def stdxo_ot_a_delta(p, alpha, testing):

        if isinstance(p, Tree):
            if testing:
                p = p.test_semantics

            else:
                p = p.train_semantics


        if which == 'first':

            return torch.mul(p, alpha) if operator == 'sum' else \
                torch.pow(p, alpha)
        else:

            return torch.mul(p, torch.sub(1, alpha)) if operator == 'sum' else \
                torch.pow(p, torch.sub(1, alpha))

    stdxo_ot_a_delta.__name__ += '_' + which


    return stdxo_ot_a_delta


def slim_alpha_geometric_crossover(operator):

    def inner_xo(p1, p2, X, X_test = None, reconstruct = True):


        alphas = [random.random() for _ in range(max(p1.size, p2.size))]

        if reconstruct:
            offs_collection = [Tree([std_xo_alpha_delta(operator=operator),
                                      p1.collection[i], p2.collection[i],
                                      alphas[i]]) for i in range(min(p1.size, p2.size))]

        offs_train_semantics = torch.stack(
            [std_xo_alpha_delta(operator=operator)(p1.train_semantics[i], p2.train_semantics[i],
                                                   alphas[i], testing = False) for i in range(min(p1.size, p2.size))])
        if p1.test_semantics is not None:
            offs_test_semantics = torch.stack(
                [std_xo_alpha_delta(operator=operator)(p1.test_semantics[i], p2.test_semantics[i],
                                                       alphas[i], testing = True) for i in range(min(p1.size, p2.size))])

        offs_nodes_collection = [p1.nodes_collection[i] + p2.nodes_collection[i] + 7
                                 for i in range(min(p1.size, p2.size))]
        offs_depth_collection = [
            max([p1.depth_collection[i] + 2, p2.depth_collection[i] + 2])
            for i in range(min(p1.size, p2.size))]

        if p1.size > p2.size:

            which = 'first'


            if reconstruct:
                offs_collection += [Tree([std_xo_alpha_ot_delta(which, operator=operator),
                          p1.collection[i], alphas[i]]) for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]

            offs_train_semantics = torch.stack([*offs_train_semantics,
                                    *[std_xo_alpha_ot_delta(which, operator=operator)(p1.train_semantics[i], alphas[i],
                                                                                      testing = False)
                                    for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]])

            if p1.test_semantics is not None:
                offs_test_semantics = torch.stack([*offs_test_semantics,
                                      *[std_xo_alpha_ot_delta(which, operator=operator)(p1.test_semantics[i], alphas[i],
                                                                                        testing = True)
                                       for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]])

            offs_nodes_collection += [p1.nodes_collection[i] + 2
                                     for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]
            offs_depth_collection += [ max([p1.depth_collection[i] + 1])
                                      for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]


        else:

            which = 'second'

            if reconstruct:
                offs_collection += [Tree([std_xo_alpha_ot_delta(which, operator=operator),
                                          p2.collection[i], alphas[i]]) for i in
                                    range(min(p1.size, p2.size), max(p1.size, p2.size))]

            offs_train_semantics = torch.stack([*offs_train_semantics,
                                                *[std_xo_alpha_ot_delta(which, operator=operator)(p2.train_semantics[i],
                                                                                                  alphas[i], testing = False)
                                                  for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]])

            if p2.test_semantics is not None:
                offs_test_semantics = torch.stack([*offs_test_semantics,
                                                   *[std_xo_alpha_ot_delta(which, operator=operator)(
                                                       p2.test_semantics[i], alphas[i], testing = True)
                                                     for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]])

            offs_nodes_collection += [p2.nodes_collection[i] + 4
                                      for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]
            offs_depth_collection += [max([p2.depth_collection[i] + 1])
                                      for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]


        offs = Individual(collection=offs_collection if reconstruct else None,
                          train_semantics=torch.Tensor(offs_train_semantics),
                          test_semantics=torch.Tensor(offs_test_semantics) if p1.test_semantics is not None else None,
                          reconstruct=reconstruct)

        offs.size = max(p1.size, p2.size)

        offs.nodes_collection = offs_nodes_collection
        offs.nodes_count = sum(offs.nodes_collection) + (offs.size - 1)

        offs.depth_collection = offs_depth_collection
        offs.depth = max([depth - (i - 1) if i != 0 else depth
                          for i, depth in enumerate(offs.depth_collection)]) + (offs.size - 1)



        return offs

    return inner_xo



def slim_swap_geometric_crossover(p1, p2, X = None, X_test = None, reconstruct = True):

    mask = [random.randint(0,1) for _ in range(max(p1.size, p2.size))]
    inv_mask = [abs(v-1) for v in mask]
    
    if reconstruct:

        off1_collection = [
            p1.collection[idx] if mask[idx] == 0 and idx < p1.size
            else p2.collection[idx]
            for idx in range(len(mask))
            if (mask[idx] == 0 and idx < p1.size) or (mask[idx] == 1 and idx < p2.size)
        ]
        
    off1_train_semantics = torch.stack([
        p1.train_semantics[idx] if mask[idx] == 0 and idx < p1.size
        else p2.train_semantics[idx]
        for idx in range(len(mask))
        if (mask[idx] == 0 and idx < p1.size) or (mask[idx] == 1 and idx < p2.size)
    ])

    if p1.test_semantics is not None:
        off1_test_semantics = torch.stack([
            p1.test_semantics[idx] if mask[idx] == 0 and idx < p1.size
            else p2.test_semantics[idx]
            for idx in range(len(mask))
            if (mask[idx] == 0 and idx < p1.size) or (mask[idx] == 1 and idx < p2.size)
        ])
    
    off1_nodes_collection = [
            p1.nodes_collection[idx] if mask[idx] == 0 and idx < p1.size
            else p2.nodes_collection[idx]
            for idx in range(len(mask))
            if (mask[idx] == 0 and idx < p1.size) or (mask[idx] == 1 and idx < p2.size)
        ]
    
    off1_depth_collection = [
            p1.depth_collection[idx] if mask[idx] == 0 and idx < p1.size
            else p2.depth_collection[idx]
            for idx in range(len(mask))
            if (mask[idx] == 0 and idx < p1.size) or (mask[idx] == 1 and idx < p2.size)
        ]

    off1 = Individual(collection=off1_collection if reconstruct else None,
                      train_semantics=torch.Tensor(off1_train_semantics),
                      test_semantics=torch.Tensor(off1_test_semantics) if p1.test_semantics is not None else None,
                      reconstruct=reconstruct)

    off1.size = len(off1.train_semantics)

    off1.nodes_collection = off1_nodes_collection
    off1.nodes_count = sum(off1.nodes_collection) + (off1.size - 1)

    off1.depth_collection = off1_depth_collection
    off1.depth = max([depth - (i - 1) if i != 0 else depth
                      for i, depth in enumerate(off1.depth_collection)]) + (off1.size - 1)

    if reconstruct:
        off2_collection = [
            p1.collection[idx] if inv_mask[idx] == 0 and idx < p1.size
            else p2.collection[idx]
            for idx in range(len(inv_mask))
            if (inv_mask[idx] == 0 and idx < p1.size) or (inv_mask[idx] == 1 and idx < p2.size)
        ]

    off2_train_semantics = torch.stack([
        p1.train_semantics[idx] if inv_mask[idx] == 0 and idx < p1.size
        else p2.train_semantics[idx]
        for idx in range(len(inv_mask))
        if (inv_mask[idx] == 0 and idx < p1.size) or (inv_mask[idx] == 1 and idx < p2.size)
    ])

    if p1.test_semantics is not None:
        off2_test_semantics = torch.stack([
            p1.test_semantics[idx] if inv_mask[idx] == 0 and idx < p1.size
            else p2.test_semantics[idx]
            for idx in range(len(inv_mask))
            if (inv_mask[idx] == 0 and idx < p1.size) or (inv_mask[idx] == 1 and idx < p2.size)
        ])

    off2_nodes_collection = [
        p1.nodes_collection[idx] if inv_mask[idx] == 0 and idx < p1.size
        else p2.nodes_collection[idx]
        for idx in range(len(inv_mask))
        if (inv_mask[idx] == 0 and idx < p1.size) or (inv_mask[idx] == 1 and idx < p2.size)
    ]

    off2_depth_collection = [
        p1.depth_collection[idx] if inv_mask[idx] == 0 and idx < p1.size
        else p2.depth_collection[idx]
        for idx in range(len(inv_mask))
        if (inv_mask[idx] == 0 and idx < p1.size) or (inv_mask[idx] == 1 and idx < p2.size)
    ]

    off2 = Individual(collection=off2_collection if reconstruct else None,
                      train_semantics=torch.Tensor(off2_train_semantics),
                      test_semantics=torch.Tensor(off2_test_semantics) if p1.test_semantics is not None else None,
                      reconstruct=reconstruct)

    off2.size = len(off2.train_semantics)

    off2.nodes_collection = off2_nodes_collection
    off2.nodes_count = sum(off2.nodes_collection) + (off2.size - 1)

    off2.depth_collection = off2_depth_collection
    off2.depth = max([depth - (i - 1) if i != 0 else depth
                      for i, depth in enumerate(off2.depth_collection)]) + (off2.size - 1)

    return off1, off2

def slim_alpha_deflate_geometric_crossover(operator, perc_off_blocks):

    def inner_xo(p1, p2, X, X_test = None, reconstruct = True):

        mask = generate_mask(max(p1.size, p2.size), int(perc_off_blocks * max(p1.size, p2.size)))

        alphas = [random.random() for _ in range(max(p1.size, p2.size)) ]

        if reconstruct:
            offs_collection = [Tree([std_xo_alpha_delta(operator=operator),
                                      p1.collection[i], p2.collection[i],
                                      alphas[i]]) for i in range(min(p1.size, p2.size)) if mask[i] == 1]

        offs_train_semantics = torch.stack(
            [std_xo_alpha_delta(operator=operator)(p1.train_semantics[i], p2.train_semantics[i],
                                                   alphas[i], testing = False) for i in range(min(p1.size, p2.size)) if mask[i] == 1])
        if p1.test_semantics is not None:
            offs_test_semantics = torch.stack(
                [std_xo_alpha_delta(operator=operator)(p1.test_semantics[i], p2.test_semantics[i],
                                                       alphas[i], testing = True) for i in range(min(p1.size, p2.size)) if mask[i] == 1])

        offs_nodes_collection = [p1.nodes_collection[i] + p2.nodes_collection[i] + + 7
                                 for i in range(min(p1.size, p2.size)) if mask[i] == 1]
        offs_depth_collection = [
            max([p1.depth_collection[i] + 2, p2.depth_collection[i] + 2])
            for i in range(min(p1.size, p2.size)) if mask[i] == 1]

        if p1.size > p2.size:

            which = 'first'


            if reconstruct:
                offs_collection += [Tree([std_xo_alpha_ot_delta(which, operator=operator),
                          p1.collection[i], alphas[i]]) for i in range(min(p1.size, p2.size), max(p1.size, p2.size)) if mask[i] == 1]

            offs_train_semantics = torch.stack([*offs_train_semantics,
                                    *[std_xo_alpha_ot_delta(which, operator=operator)(p1.train_semantics[i], alphas[i], testing = False)
                                    for i in range(min(p1.size, p2.size), max(p1.size, p2.size)) if mask[i] == 1]])

            if p1.test_semantics is not None:
                offs_test_semantics = torch.stack([*offs_test_semantics,
                                      *[std_xo_alpha_ot_delta(which, operator=operator)(p1.test_semantics[i], alphas[i], testing = True)
                                       for i in range(min(p1.size, p2.size), max(p1.size, p2.size)) if mask[i] == 1]])

            offs_nodes_collection += [p1.nodes_collection[i] + 2
                                     for i in range(min(p1.size, p2.size), max(p1.size, p2.size)) if mask[i] == 1]
            offs_depth_collection += [ max([p1.depth_collection[i] + 1])
                                      for i in range(min(p1.size, p2.size), max(p1.size, p2.size)) if mask[i] == 1]


        else:

            which = 'second'

            if reconstruct:
                offs_collection += [Tree([std_xo_alpha_ot_delta(which, operator=operator),
                                          p2.collection[i], alphas[i]]) for i in
                                    range(min(p1.size, p2.size), max(p1.size, p2.size)) if mask[i] == 1]

            offs_train_semantics = torch.stack([*offs_train_semantics,
                                                *[std_xo_alpha_ot_delta(which, operator=operator)(p2.train_semantics[i],
                                                                                                  alphas[i], testing = False)
                                                  for i in range(min(p1.size, p2.size), max(p1.size, p2.size)) if mask[i] == 1]])

            if p2.test_semantics is not None:
                offs_test_semantics = torch.stack([*offs_test_semantics,
                                                   *[std_xo_alpha_ot_delta(which, operator=operator)(
                                                       p2.test_semantics[i], alphas[i], testing = True)
                                                     for i in range(min(p1.size, p2.size), max(p1.size, p2.size)) if mask[i] == 1]])

            offs_nodes_collection += [p2.nodes_collection[i] + 4
                                      for i in range(min(p1.size, p2.size), max(p1.size, p2.size)) if mask[i] == 1]
            offs_depth_collection += [max([p2.depth_collection[i] + 1])
                                      for i in range(min(p1.size, p2.size), max(p1.size, p2.size)) if mask[i] == 1]


        offs = Individual(collection=offs_collection if reconstruct else None,
                          train_semantics=torch.Tensor(offs_train_semantics),
                          test_semantics=torch.Tensor(offs_test_semantics) if p1.test_semantics is not None else None,
                          reconstruct=reconstruct)

        offs.size = len(offs.train_semantics)

        offs.nodes_collection = offs_nodes_collection
        offs.nodes_count = sum(offs.nodes_collection) + (offs.size - 1)

        offs.depth_collection = offs_depth_collection
        offs.depth = max([depth - (i - 1) if i != 0 else depth
                          for i, depth in enumerate(offs.depth_collection)]) + (offs.size - 1)



        return offs

    return inner_xo


def slim_swap_deflate_geometric_crossover(perc_off_blocks):
    
    def inner_ssd_gxo(p1, p2, X = None, X_test = None, reconstruct = True):

        mask_selection = generate_mask(max(p1.size, p2.size), int(perc_off_blocks * max(p1.size, p2.size)))
        
        mask_parents = [random.randint(0,1) for _ in range(max(p1.size, p2.size))]
        inv_mask_parents = [abs(v-1) for v in mask_parents]

        if reconstruct:
            off1_collection = [
                p1.collection[idx] if mask_parents[idx] == 0 and idx < p1.size
                else p2.collection[idx]
                for idx in range(len(mask_parents))
                if ((mask_parents[idx] == 0 and idx < p1.size) or (mask_parents[idx] == 1 and idx < p2.size))
                and mask_selection[idx] == 1
            ]

        off1_train_semantics = torch.stack([
            p1.train_semantics[idx] if mask_parents[idx] == 0 and idx < p1.size
            else p2.train_semantics[idx]
            for idx in range(len(mask_parents))
            if ((mask_parents[idx] == 0 and idx < p1.size) or (mask_parents[idx] == 1 and idx < p2.size))
            and mask_selection[idx] == 1
        ])
        if p1.test_semantics is not None:
            off1_test_semantics = torch.stack([
                p1.test_semantics[idx] if mask_parents[idx] == 0 and idx < p1.size
                else p2.test_semantics[idx]
                for idx in range(len(mask_parents))
                if ((mask_parents[idx] == 0 and idx < p1.size) or (mask_parents[idx] == 1 and idx < p2.size))
                and mask_selection[idx] == 1
            ])

        off1_nodes_collection = [
            p1.nodes_collection[idx] if mask_parents[idx] == 0 and idx < p1.size
            else p2.nodes_collection[idx]
            for idx in range(len(mask_parents))
            if ((mask_parents[idx] == 0 and idx < p1.size) or (mask_parents[idx] == 1 and idx < p2.size))
            and mask_selection[idx] == 1
        ]

        off1_depth_collection = [
            p1.depth_collection[idx] if mask_parents[idx] == 0 and idx < p1.size
            else p2.depth_collection[idx]
            for idx in range(len(mask_parents))
            if ((mask_parents[idx] == 0 and idx < p1.size) or (mask_parents[idx] == 1 and idx < p2.size))
            and mask_selection[idx] == 1
        ]

        off1 = Individual(collection=off1_collection if reconstruct else None,
                          train_semantics=torch.Tensor(off1_train_semantics),
                          test_semantics=torch.Tensor(off1_test_semantics) if p1.test_semantics is not None else None,
                          reconstruct=reconstruct)

        off1.size = len(off1.train_semantics)

        off1.nodes_collection = off1_nodes_collection
        off1.nodes_count = sum(off1.nodes_collection) + (off1.size - 1)

        off1.depth_collection = off1_depth_collection
        off1.depth = max([depth - (i - 1) if i != 0 else depth
                          for i, depth in enumerate(off1.depth_collection)]) + (off1.size - 1)

        if reconstruct:
            off2_collection = [
                p1.collection[idx] if inv_mask_parents[idx] == 0 and idx < p1.size
                else p2.collection[idx]
                for idx in range(len(inv_mask_parents))
                if ((inv_mask_parents[idx] == 0 and idx < p1.size) or (inv_mask_parents[idx] == 1 and idx < p2.size))
                and mask_selection[idx] == 1
            ]

        off2_train_semantics = torch.stack([
            p1.train_semantics[idx] if inv_mask_parents[idx] == 0 and idx < p1.size
            else p2.train_semantics[idx]
            for idx in range(len(inv_mask_parents))
            if ((inv_mask_parents[idx] == 0 and idx < p1.size) or (inv_mask_parents[idx] == 1 and idx < p2.size))
            and mask_selection[idx] == 1
        ])

        if p1.test_semantics is not None:
            off2_test_semantics = torch.stack([
                p1.test_semantics[idx] if inv_mask_parents[idx] == 0 and idx < p1.size
                else p2.test_semantics[idx]
                for idx in range(len(inv_mask_parents))
                if ((inv_mask_parents[idx] == 0 and idx < p1.size) or (inv_mask_parents[idx] == 1 and idx < p2.size))
                and mask_selection[idx] == 1
            ])

        off2_nodes_collection = [
            p1.nodes_collection[idx] if inv_mask_parents[idx] == 0 and idx < p1.size
            else p2.nodes_collection[idx]
            for idx in range(len(inv_mask_parents))
            if ((inv_mask_parents[idx] == 0 and idx < p1.size) or (inv_mask_parents[idx] == 1 and idx < p2.size))
            and mask_selection[idx] == 1
        ]

        off2_depth_collection = [
            p1.depth_collection[idx] if inv_mask_parents[idx] == 0 and idx < p1.size
            else p2.depth_collection[idx]
            for idx in range(len(inv_mask_parents))
            if ((inv_mask_parents[idx] == 0 and idx < p1.size) or (inv_mask_parents[idx] == 1 and idx < p2.size))
            and mask_selection[idx] == 1
        ]

        off2 = Individual(collection=off2_collection if reconstruct else None,
                          train_semantics=torch.Tensor(off2_train_semantics),
                          test_semantics=torch.Tensor(off2_test_semantics) if p1.test_semantics is not None else None,
                          reconstruct=reconstruct)

        off2.size = len(off2.train_semantics)

        off2.nodes_collection = off2_nodes_collection
        off2.nodes_count = sum(off2.nodes_collection) + (off2.size - 1)

        off2.depth_collection = off2_depth_collection
        off2.depth = max([depth - (i - 1) if i != 0 else depth
                          for i, depth in enumerate(off2.depth_collection)]) + (off2.size - 1)
    
        return off1, off2
    
    return inner_ssd_gxo

def donor_gxo(p1, p2,  X = None, X_test = None, reconstruct = True):

    #choose at random the index of the donor (0 = p1, 1= p2)
    donor_idx = random.randint(0,1)
    donor = p1 if donor_idx == 0 else p2
    recipient = p1 if donor_idx == 1 else p2

    #choose at random a block from the donor
    if donor.size > 1:
        donation_idx = random.randint(1, donor.size - 1 )

        recipient_offs = Individual(collection=[*recipient.collection, donor[donation_idx]]
                         if reconstruct else None,
                          train_semantics=torch.stack([*recipient.train_semantics, donor.train_semantics[donation_idx]]),
                          test_semantics=torch.stack([*recipient.test_semantics, donor.test_semantics[donation_idx]])
                          if donor.test_semantics is not None
                          else None,
                          reconstruct=reconstruct)

        donor_offs = Individual(collection=donor.collection[:donation_idx] + donor.collection[donation_idx+1:]
                         if reconstruct else None,
                          train_semantics=torch.stack(
                                [*donor.train_semantics[:donation_idx], *donor.train_semantics[donation_idx + 1:]]),
                          test_semantics=torch.stack(
                [*donor.test_semantics[:donation_idx], *donor.test_semantics[donation_idx + 1:]])
                          if donor.test_semantics is not None
                          else None,
                          reconstruct=reconstruct)

        recipient_offs.nodes_collection = recipient.nodes_collection + [donor.nodes_collection[donation_idx]]
        donor_offs.nodes_collection = donor.nodes_collection[:donation_idx] + donor.nodes_collection[donation_idx + 1:]

        recipient_offs.depth_collection = recipient.depth_collection + [donor.depth_collection[donation_idx]]
        donor_offs.depth_collection = donor.depth_collection[:donation_idx] + donor.depth_collection[donation_idx + 1:]

        recipient_offs.size = recipient.size + 1
        donor_offs.size = donor.size - 1

        recipient_offs.nodes_count = sum(recipient_offs.nodes_collection) + (recipient_offs.size - 1)
        donor_offs.nodes_count = sum(donor_offs.nodes_collection) + (donor_offs.size - 1)

        recipient_offs.depth = sum(recipient_offs.depth_collection) + (recipient_offs.size - 1)
        donor_offs.depth = sum(donor_offs.depth_collection) + (donor_offs.size - 1)

        return donor_offs, recipient_offs

    else:

        return p1, p2

def mb_donor_gxo(donor_perc = None):

    def mbdgxo(p1, p2,  X = None, X_test = None, reconstruct = True):

        # choose at random the index of the donor (0 = p1, 1= p2)
        donor_idx = random.randint(0, 1)
        donor = p1 if donor_idx == 0 else p2
        recipient = p1 if donor_idx == 1 else p2

        n_blocks = round(donor_perc * donor.size)

        # choose at random a block from the donor
        if donor.size > 1 and n_blocks > 1:

            for _ in range(n_blocks):

                donation_idx = random.randint(1, donor.size - 1)

                if reconstruct:
                    recipient.collection = recipient.collection + [donor.collection[donation_idx]]
                    donor.collection.pop[donor_idx]

                recipient.train_semantics = torch.concatenate(
                    (recipient.train_semantics, donor.train_semantics[donation_idx].unsqueeze(0)), dim=0)
                donor.train_semantics = torch.cat(
                    (donor.train_semantics[:donation_idx], donor.train_semantics[donation_idx + 1:]), dim=0)

                if donor.test_semantics is not None:
                    recipient.test_semantics = torch.concatenate(
                        (recipient.test_semantics, donor.test_semantics[donation_idx].unsqueeze(0)), dim=0)
                    donor.test_semantics = torch.cat(
                        (donor.test_semantics[:donation_idx], donor.test_semantics[donation_idx + 1:]), dim=0)

                recipient.size += 1
                donor.size -= 1

                recipient.nodes_collection = recipient.nodes_collection + [donor.nodes_collection[donation_idx]]
                donor.nodes_collection.pop(donation_idx)

                recipient.nodes_count = sum(recipient.nodes_collection) + (recipient.size - 1)
                donor.nodes_count = sum(donor.nodes_collection) + (donor.size - 1)

                recipient.depth_collection = recipient.depth_collection + [donor.depth_collection[donation_idx]]
                donor.depth_collection.pop(donation_idx)

                recipient.depth = sum(recipient.depth_collection) + (recipient.size - 1)
                donor.depth = sum(donor.depth_collection) + (donor.size - 1)

            return donor, recipient

        else:

            return p1, p2


    return mbdgxo

