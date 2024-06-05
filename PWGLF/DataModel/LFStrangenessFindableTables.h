// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <cmath>
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "Common/Core/RecoDecay.h"
#include "CommonConstants/PhysicsConstants.h"

#ifndef PWGLF_DATAMODEL_LFSTRANGENESSFINDABLETABLES_H_
#define PWGLF_DATAMODEL_LFSTRANGENESSFINDABLETABLES_H_

using namespace o2;
using namespace o2::framework;

// Creating output TTree for ML analysis
namespace o2::aod
{
namespace v0findcand
{
DECLARE_SOA_COLUMN(PosITSCls, posITSCls, int);
DECLARE_SOA_COLUMN(NegITSCls, negITSCls, int);
DECLARE_SOA_COLUMN(PosITSClSize, posITSClSize, uint32_t);
DECLARE_SOA_COLUMN(NegITSClSize, negITSClSize, uint32_t);
DECLARE_SOA_COLUMN(PosTPCRows, posTPCRows, uint8_t);
DECLARE_SOA_COLUMN(NegTPCRows, negTPCRows, uint8_t);
DECLARE_SOA_COLUMN(PosTPCSigmaPi, posTPCSigmaPi, float);
DECLARE_SOA_COLUMN(NegTPCSigmaPi, negTPCSigmaPi, float);
DECLARE_SOA_COLUMN(PosTPCSigmaPr, posTPCSigmaPr, float);
DECLARE_SOA_COLUMN(NegTPCSigmaPr, negTPCSigmaPr, float);
DECLARE_SOA_COLUMN(PosTPCSigmaEl, posTPCSigmaEl, float);
DECLARE_SOA_COLUMN(NegTPCSigmaEl, negTPCSigmaEl, float);
DECLARE_SOA_COLUMN(PxPos, pxpos, float); //! positive track px at min
DECLARE_SOA_COLUMN(PyPos, pypos, float); //! positive track py at min
DECLARE_SOA_COLUMN(PzPos, pzpos, float); //! positive track pz at min
DECLARE_SOA_COLUMN(PxNeg, pxneg, float); //! negative track px at min
DECLARE_SOA_COLUMN(PyNeg, pyneg, float); //! negative track py at min
DECLARE_SOA_COLUMN(PzNeg, pzneg, float); //! negative track pz at min
DECLARE_SOA_COLUMN(X, x, float);         //! decay position X
DECLARE_SOA_COLUMN(Y, y, float);         //! decay position Y
DECLARE_SOA_COLUMN(Z, z, float);         //! decay position Z
DECLARE_SOA_COLUMN(PT, pT, float);
DECLARE_SOA_COLUMN(PosEta, posEta, float);
DECLARE_SOA_COLUMN(NegEta, negEta, float);
DECLARE_SOA_COLUMN(V0Eta, v0Eta, float);
DECLARE_SOA_COLUMN(V0radius, v0radius, float);
DECLARE_SOA_COLUMN(PA, pa, float);
DECLARE_SOA_COLUMN(DCApostopv, dcapostopv, float);
DECLARE_SOA_COLUMN(DCAnegtopv, dcanegtopv, float);
DECLARE_SOA_COLUMN(DCAV0daughters, dcaV0daughters, float);
DECLARE_SOA_COLUMN(DCAv0topv, dcav0topv, float);
DECLARE_SOA_COLUMN(PsiPair, psiPair, float);
DECLARE_SOA_COLUMN(Centrality, centrality, float);
DECLARE_SOA_COLUMN(PDGCodeMother, pdgCodeMother, int);
DECLARE_SOA_COLUMN(V0ID, v0ID, int);
DECLARE_SOA_COLUMN(RecoStatus, recoStatus, bool); // is acceptably track or found 
} // namespace v0findcand

DECLARE_SOA_TABLE(V0FindCands, "AOD", "V0FINDCAND",
                  v0findcand::PosITSCls,
                  v0findcand::NegITSCls,
                  v0findcand::PosITSClSize,
                  v0findcand::NegITSClSize,
                  v0findcand::PosTPCRows,
                  v0findcand::NegTPCRows,
                  v0findcand::PosTPCSigmaPi,
                  v0findcand::NegTPCSigmaPi,
                  v0findcand::PosTPCSigmaPr,
                  v0findcand::NegTPCSigmaPr,
                  v0findcand::PosTPCSigmaEl,
                  v0findcand::NegTPCSigmaEl,
                  v0findcand::PxPos,
                  v0findcand::PyPos,
                  v0findcand::PzPos,
                  v0findcand::PxNeg,
                  v0findcand::PyNeg,
                  v0findcand::PzNeg,
                  v0findcand::X,
                  v0findcand::Y,
                  v0findcand::Z,
                  v0findcand::PT,
                  v0findcand::PosEta,
                  v0findcand::NegEta,
                  v0findcand::V0Eta,
                  v0findcand::V0radius,
                  v0findcand::PA,
                  v0findcand::DCApostopv,
                  v0findcand::DCAnegtopv,
                  v0findcand::DCAV0daughters,
                  v0findcand::DCAv0topv,
                  v0findcand::PsiPair,
                  v0findcand::Centrality,
                  v0findcand::PDGCodeMother,
                  v0findcand::V0ID,
                  v0findcand::RecoStatus);


} // namespace o2::aod

#endif // PWGLF_DATAMODEL_LFSTRANGENESSFINDABLETABLES_H_
