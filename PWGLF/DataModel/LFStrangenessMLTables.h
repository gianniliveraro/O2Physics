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

#ifndef PWGLF_DATAMODEL_LFSTRANGENESSMLTABLES_H_
#define PWGLF_DATAMODEL_LFSTRANGENESSMLTABLES_H_

// Creating output TTree for ML analysis
namespace o2::aod
{
namespace v0mlcandidates
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
DECLARE_SOA_COLUMN(TOFSigmaLaPr, tofSigmaLaPr, float);
DECLARE_SOA_COLUMN(TOFSigmaLaPi, tofSigmaLaPi, float);
DECLARE_SOA_COLUMN(TOFSigmaALaPi, tofSigmaALaPi, float);
DECLARE_SOA_COLUMN(TOFSigmaALaPr, tofSigmaALaPr, float);
DECLARE_SOA_COLUMN(TOFSigmaK0PiPlus, tofSigmaK0PiPlus, float);
DECLARE_SOA_COLUMN(TOFSigmaK0PiMinus, tofSigmaK0PiMinus, float);
DECLARE_SOA_COLUMN(LambdaMass, lambdaMass, float);
DECLARE_SOA_COLUMN(AntiLambdaMass, antiLambdaMass, float);
DECLARE_SOA_COLUMN(GammaMass, gammaMass, float);
DECLARE_SOA_COLUMN(KZeroShortMass, kZeroShortMass, float);
DECLARE_SOA_COLUMN(PT, pT, float);
DECLARE_SOA_COLUMN(Qt, qt, float);
DECLARE_SOA_COLUMN(Alpha, alpha, float);
DECLARE_SOA_COLUMN(PosEta, posEta, float);
DECLARE_SOA_COLUMN(NegEta, negEta, float);
DECLARE_SOA_COLUMN(V0Eta, v0Eta, float);
DECLARE_SOA_COLUMN(Z, z, float);
DECLARE_SOA_COLUMN(V0radius, v0radius, float);
DECLARE_SOA_COLUMN(PA, pa, float);
DECLARE_SOA_COLUMN(DCApostopv, dcapostopv, float);
DECLARE_SOA_COLUMN(DCAnegtopv, dcanegtopv, float);
DECLARE_SOA_COLUMN(DCAV0daughters, dcaV0daughters, float);
DECLARE_SOA_COLUMN(DCAv0topv, dcav0topv, float);
DECLARE_SOA_COLUMN(PsiPair, psiPair, float);
DECLARE_SOA_COLUMN(V0type, v0type, uint8_t);
DECLARE_SOA_COLUMN(Centrality, centrality, float);
DECLARE_SOA_COLUMN(SelHypothesis, selHypothesis, uint8_t);
DECLARE_SOA_COLUMN(IsLambda, isLambda, bool);
DECLARE_SOA_COLUMN(IsAntiLambda, isAntiLambda, bool);
DECLARE_SOA_COLUMN(IsGamma, isGamma, bool);
DECLARE_SOA_COLUMN(IsKZeroShort, isKZeroShort, bool);
DECLARE_SOA_COLUMN(PDGCodeMother, pdgCodeMother, int);
} // namespace v0mlcandidates

DECLARE_SOA_TABLE(V0MLCandidates, "AOD", "V0MLCANDIDATES",
                  v0mlcandidates::PosITSCls,
                  v0mlcandidates::NegITSCls,
                  v0mlcandidates::PosITSClSize,
                  v0mlcandidates::NegITSClSize,
                  v0mlcandidates::PosTPCRows,
                  v0mlcandidates::NegTPCRows,
                  v0mlcandidates::PosTPCSigmaPi,
                  v0mlcandidates::NegTPCSigmaPi,
                  v0mlcandidates::PosTPCSigmaPr,
                  v0mlcandidates::NegTPCSigmaPr,
                  v0mlcandidates::PosTPCSigmaEl,
                  v0mlcandidates::NegTPCSigmaEl,
                  v0mlcandidates::TOFSigmaLaPr,
                  v0mlcandidates::TOFSigmaLaPi,
                  v0mlcandidates::TOFSigmaALaPi,
                  v0mlcandidates::TOFSigmaALaPr,
                  v0mlcandidates::TOFSigmaK0PiPlus,
                  v0mlcandidates::TOFSigmaK0PiMinus,
                  v0mlcandidates::LambdaMass,
                  v0mlcandidates::AntiLambdaMass,
                  v0mlcandidates::GammaMass,
                  v0mlcandidates::KZeroShortMass,
                  v0mlcandidates::PT,
                  v0mlcandidates::Qt,
                  v0mlcandidates::Alpha,
                  v0mlcandidates::PosEta,
                  v0mlcandidates::NegEta,
                  v0mlcandidates::V0Eta,
                  v0mlcandidates::Z,
                  v0mlcandidates::V0radius,
                  v0mlcandidates::PA,
                  v0mlcandidates::DCApostopv,
                  v0mlcandidates::DCAnegtopv,
                  v0mlcandidates::DCAV0daughters,
                  v0mlcandidates::DCAv0topv,
                  v0mlcandidates::PsiPair,
                  v0mlcandidates::V0type,
                  v0mlcandidates::Centrality,
                  v0mlcandidates::SelHypothesis,
                  v0mlcandidates::IsLambda,
                  v0mlcandidates::IsAntiLambda,
                  v0mlcandidates::IsGamma,
                  v0mlcandidates::IsKZeroShort,
                  v0mlcandidates::PDGCodeMother);

namespace V0MLSelection
{
DECLARE_SOA_COLUMN(GammaBDTScore, gammaBDTScore, float);
DECLARE_SOA_COLUMN(LambdaBDTScore, lambdaBDTScore, float);
DECLARE_SOA_COLUMN(AntiLambdaBDTScore, antiLambdaBDTScore, float);
DECLARE_SOA_COLUMN(K0ShortBDTScore, k0ShortBDTScore, float);
} // namespace V0MLSelection

DECLARE_SOA_TABLE(V0GammaMLScores, "AOD", "V0GaMLScores",
                  V0MLSelection::GammaBDTScore);
DECLARE_SOA_TABLE(V0LambdaMLScores, "AOD", "V0LaMLScores",
                  V0MLSelection::LambdaBDTScore);
DECLARE_SOA_TABLE(V0AntiLambdaMLScores, "AOD", "V0ALaMLScores",
                  V0MLSelection::AntiLambdaBDTScore);
DECLARE_SOA_TABLE(V0K0ShortMLScores, "AOD", "V0K0MLScores",
                  V0MLSelection::K0ShortBDTScore);

// Cascade candidates
namespace cascmlcandidates
{
DECLARE_SOA_COLUMN(MassWindows, massWindows, uint8_t);
DECLARE_SOA_COLUMN(Charge, charge, int);
DECLARE_SOA_COLUMN(Centrality, centrality, float);
DECLARE_SOA_COLUMN(PosITSCls, posITSCls, int);
DECLARE_SOA_COLUMN(NegITSCls, negITSCls, int);
DECLARE_SOA_COLUMN(BachITSCls, bachITSCls, int);
DECLARE_SOA_COLUMN(PosITSClSize, posITSClSize, int);
DECLARE_SOA_COLUMN(NegITSClSize, negITSClSize, int);
DECLARE_SOA_COLUMN(BachITSClSize, bachITSClSize, int);
DECLARE_SOA_COLUMN(PosTPCRows, posTPCRows, float);
DECLARE_SOA_COLUMN(NegTPCRows, negTPCRows, float);
DECLARE_SOA_COLUMN(BachTPCRows, bachTPCRows, float);
DECLARE_SOA_COLUMN(PosTPCSigmaPi, posTPCSigmaPi, float);
DECLARE_SOA_COLUMN(NegTPCSigmaPi, negTPCSigmaPi, float);
DECLARE_SOA_COLUMN(PosTPCSigmaPr, posTPCSigmaPr, float);
DECLARE_SOA_COLUMN(NegTPCSigmaPr, negTPCSigmaPr, float);
DECLARE_SOA_COLUMN(BachTPCSigmaPi, bachTPCSigmaPi, float);
DECLARE_SOA_COLUMN(BachTPCSigmaKa, bachTPCSigmaKa, float);
DECLARE_SOA_COLUMN(TOFNSigmaXiLaPi, tofNSigmaXiLaPi, float);
DECLARE_SOA_COLUMN(TOFNSigmaXiLaPr, tofNSigmaXiLaPr, float);
DECLARE_SOA_COLUMN(TOFNSigmaXiPi, tofNSigmaXiPi, float);
DECLARE_SOA_COLUMN(TOFNSigmaOmLaPi, tofNSigmaOmLaPi, float);
DECLARE_SOA_COLUMN(TOFNSigmaOmLaPr, tofNSigmaOmLaPr, float);
DECLARE_SOA_COLUMN(TOFNSigmaOmKa, tofNSigmaOmKa, float);

DECLARE_SOA_COLUMN(MXi, mXi, float);
DECLARE_SOA_COLUMN(MOmega, mOmega, float);
DECLARE_SOA_COLUMN(YXi, yXi, float);
DECLARE_SOA_COLUMN(YOmega, yOmega, float);
DECLARE_SOA_COLUMN(MLambda, mLambda, float);
DECLARE_SOA_COLUMN(Pt, pt, float);
DECLARE_SOA_COLUMN(PosEta, posEta, float);
DECLARE_SOA_COLUMN(NegEta, negEta, float);
DECLARE_SOA_COLUMN(BachEta, bachEta, float);

DECLARE_SOA_COLUMN(V0radius, v0radius, float);
DECLARE_SOA_COLUMN(CascRadius, cascradius, float);
DECLARE_SOA_COLUMN(DCApostopv, dcapostopv, float);
DECLARE_SOA_COLUMN(DCAnegtopv, dcanegtopv, float);
DECLARE_SOA_COLUMN(DCAbachtopv, dcabachtopv, float);
DECLARE_SOA_COLUMN(DCAV0daughters, dcaV0daughters, float);
DECLARE_SOA_COLUMN(DCACascDaughters, dcaCascDaughters, float);
DECLARE_SOA_COLUMN(DCAv0topv, dcav0topv, float);
DECLARE_SOA_COLUMN(V0PA, v0PA, float);
DECLARE_SOA_COLUMN(CascPA, cascPA, float);

DECLARE_SOA_COLUMN(IsXiMinus, isXiMinus, bool);
DECLARE_SOA_COLUMN(IsXiPlus, isXiPlus, bool);
DECLARE_SOA_COLUMN(IsOmegaMinus, isOmegaMinus, bool);
DECLARE_SOA_COLUMN(IsOmegaPlus, isOmegaPlus, bool);
} // namespace cascmlcandidates

DECLARE_SOA_TABLE(CascMLCandidates, "AOD", "CAMLCANDIDATES",
                  cascmlcandidates::MassWindows,
                  cascmlcandidates::Charge,
                  cascmlcandidates::Centrality,
                  cascmlcandidates::PosITSCls,
                  cascmlcandidates::NegITSCls,
                  cascmlcandidates::BachITSCls,
                  cascmlcandidates::PosITSClSize,
                  cascmlcandidates::NegITSClSize,
                  cascmlcandidates::BachITSClSize,
                  cascmlcandidates::PosTPCRows,
                  cascmlcandidates::NegTPCRows,
                  cascmlcandidates::BachTPCRows,
                  cascmlcandidates::PosTPCSigmaPi,
                  cascmlcandidates::NegTPCSigmaPi,
                  cascmlcandidates::PosTPCSigmaPr,
                  cascmlcandidates::NegTPCSigmaPr,
                  cascmlcandidates::BachTPCSigmaPi,
                  cascmlcandidates::BachTPCSigmaKa,
                  cascmlcandidates::TOFNSigmaXiLaPi,
                  cascmlcandidates::TOFNSigmaXiLaPr,
                  cascmlcandidates::TOFNSigmaXiPi,
                  cascmlcandidates::TOFNSigmaOmLaPi,
                  cascmlcandidates::TOFNSigmaOmLaPr,
                  cascmlcandidates::TOFNSigmaOmKa,
                  cascmlcandidates::MXi,
                  cascmlcandidates::MOmega,
                  cascmlcandidates::YXi,
                  cascmlcandidates::YOmega,
                  cascmlcandidates::MLambda,
                  cascmlcandidates::Pt,
                  cascmlcandidates::PosEta,
                  cascmlcandidates::NegEta,
                  cascmlcandidates::BachEta,
                  cascmlcandidates::V0radius,
                  cascmlcandidates::CascRadius,
                  cascmlcandidates::DCApostopv,
                  cascmlcandidates::DCAnegtopv,
                  cascmlcandidates::DCAbachtopv,
                  cascmlcandidates::DCAV0daughters,
                  cascmlcandidates::DCACascDaughters,
                  cascmlcandidates::DCAv0topv,
                  cascmlcandidates::V0PA,
                  cascmlcandidates::CascPA,
                  cascmlcandidates::IsXiMinus,
                  cascmlcandidates::IsXiPlus,
                  cascmlcandidates::IsOmegaMinus,
                  cascmlcandidates::IsOmegaPlus);

namespace CascMLSelection
{
DECLARE_SOA_COLUMN(XiBDTScore, xiBDTScore, float);
DECLARE_SOA_COLUMN(OmegaBDTScore, omegaBDTScore, float);
} // namespace CascMLSelection

DECLARE_SOA_TABLE(CascXiMLScores, "AOD", "CASCXIMLSCORES",
                  CascMLSelection::XiBDTScore);
DECLARE_SOA_TABLE(CascOmMLScores, "AOD", "CASCOMMLSCORES",
                  CascMLSelection::OmegaBDTScore);

        
namespace PhotonDuplicates
{
DECLARE_SOA_COLUMN(V0z, v0z, float);
DECLARE_SOA_COLUMN(V0DCADau, v0DCADau, float);
DECLARE_SOA_COLUMN(V0DCADauxy, v0DCADauxy, float);
DECLARE_SOA_COLUMN(V0DCADauz, v0DCADauz, float);
DECLARE_SOA_COLUMN(V0PA, v0PA, float);
DECLARE_SOA_COLUMN(V0Radius, v0Radius, float);
DECLARE_SOA_COLUMN(V0PosDCAToPV, v0PosDCAToPV, float);
DECLARE_SOA_COLUMN(V0NegDCAToPV, v0NegDCAToPV, float);
DECLARE_SOA_COLUMN(V0DCAToPVxy, v0DCAToPVxy, float);
DECLARE_SOA_COLUMN(V0DCAToPVz, v0DCAToPVz, float);
DECLARE_SOA_COLUMN(V0Phi, v0Phi, float);
DECLARE_SOA_COLUMN(V0Collx, v0Collx, float);
DECLARE_SOA_COLUMN(V0Colly, v0Colly, float);
DECLARE_SOA_COLUMN(V0Collz, v0Collz, float);
DECLARE_SOA_COLUMN(V0AvgDCADauxy, v0AvgDCADauxy, float);
DECLARE_SOA_COLUMN(V0AvgDCADauz, v0AvgDCADauz, float);
DECLARE_SOA_COLUMN(V0AvgPA, v0AvgPA, float);
DECLARE_SOA_COLUMN(V0Avgz, v0Avgz, float);
DECLARE_SOA_COLUMN(V0MinPA, v0MinPA, float);
DECLARE_SOA_COLUMN(V0MaxPA, v0MaxPA, float);
DECLARE_SOA_COLUMN(V0MinZ, v0MinZ, float);
DECLARE_SOA_COLUMN(V0MaxZ, v0MaxZ, float);
DECLARE_SOA_COLUMN(V0MinDCADauxy, v0MinDCADauxy, float);
DECLARE_SOA_COLUMN(V0MaxDCADauxy, v0MaxDCADauxy, float);
DECLARE_SOA_COLUMN(V0MinDCADauz, v0MinDCADauz, float);
DECLARE_SOA_COLUMN(V0MaxDCADauz, v0MaxDCADauz, float);
DECLARE_SOA_COLUMN(V0MinV0DCAxy, v0MinV0DCAxy, float);
DECLARE_SOA_COLUMN(V0MaxV0DCAxy, v0MaxV0DCAxy, float);
DECLARE_SOA_COLUMN(V0MinV0DCAz, v0MinV0DCAz, float);
DECLARE_SOA_COLUMN(V0MaxV0DCAz, v0MaxV0DCAz, float);
DECLARE_SOA_COLUMN(V0PARank, v0PARank, int);
DECLARE_SOA_COLUMN(V0ZRank, v0ZRank, int);
DECLARE_SOA_COLUMN(V0DCADauxyRank, v0DCADauxyRank, int);
DECLARE_SOA_COLUMN(V0DCADauzRank, v0DCADauzRank, int);
DECLARE_SOA_COLUMN(V0DCAxyRank, v0DCAxyRank, int);
DECLARE_SOA_COLUMN(V0DCAzRank, v0DCAzRank, int);
DECLARE_SOA_COLUMN(V0PhotonMass, v0PhotonMass, float);
DECLARE_SOA_COLUMN(V0pt, v0pt, float);
DECLARE_SOA_COLUMN(V0px, v0px, float);
DECLARE_SOA_COLUMN(V0py, v0py, float);
DECLARE_SOA_COLUMN(V0pz, v0pz, float);
DECLARE_SOA_COLUMN(V0Y, v0Y, float);
DECLARE_SOA_COLUMN(V0Eta, v0Eta, float);
DECLARE_SOA_COLUMN(V0PosTrackTime, v0PosTrackTime, float);
DECLARE_SOA_COLUMN(V0NegTrackTime, v0NegTrackTime, float);
DECLARE_SOA_COLUMN(V0CollTime, v0CollTime, float);
DECLARE_SOA_COLUMN(V0NDuplicates, v0NDuplicates, int);
DECLARE_SOA_COLUMN(V0GroupID, v0GroupID, int);
DECLARE_SOA_COLUMN(V0PDGCode, v0PDGCode, int);
DECLARE_SOA_COLUMN(IsCorrectlyAssoc, isCorrectlyAssoc, bool);
} // namespace PhotonDuplicates

DECLARE_SOA_TABLE(V0Duplicates, "AOD", "V0DUPLICATES",
                  PhotonDuplicates::V0z,
                  PhotonDuplicates::V0DCADau,
                  PhotonDuplicates::V0DCADauxy,
                  PhotonDuplicates::V0DCADauz,
                  PhotonDuplicates::V0PA,
                  PhotonDuplicates::V0Radius,                  
                  PhotonDuplicates::V0PosDCAToPV,
                  PhotonDuplicates::V0NegDCAToPV,
                  PhotonDuplicates::V0DCAToPVxy,
                  PhotonDuplicates::V0DCAToPVz,
                  PhotonDuplicates::V0Phi,                  
                  PhotonDuplicates::V0Collx,
                  PhotonDuplicates::V0Colly,
                  PhotonDuplicates::V0Collz,
                  PhotonDuplicates::V0AvgDCADauxy,
                  PhotonDuplicates::V0AvgDCADauz,
                  PhotonDuplicates::V0AvgPA,
                  PhotonDuplicates::V0Avgz,
                  PhotonDuplicates::V0MinPA,
                  PhotonDuplicates::V0MaxPA,
                  PhotonDuplicates::V0MinZ,
                  PhotonDuplicates::V0MaxZ,
                  PhotonDuplicates::V0MinDCADauxy,
                  PhotonDuplicates::V0MaxDCADauxy,
                  PhotonDuplicates::V0MinDCADauz,
                  PhotonDuplicates::V0MaxDCADauz,
                  PhotonDuplicates::V0MinV0DCAxy,
                  PhotonDuplicates::V0MaxV0DCAxy,
                  PhotonDuplicates::V0MinV0DCAz,
                  PhotonDuplicates::V0MaxV0DCAz,
                  PhotonDuplicates::V0PARank,
                  PhotonDuplicates::V0ZRank,
                  PhotonDuplicates::V0DCADauxyRank,
                  PhotonDuplicates::V0DCADauzRank,
                  PhotonDuplicates::V0DCAxyRank,
                  PhotonDuplicates::V0DCAzRank,
                  PhotonDuplicates::V0PhotonMass,
                  PhotonDuplicates::V0pt,
                  PhotonDuplicates::V0px,
                  PhotonDuplicates::V0py,
                  PhotonDuplicates::V0pz,
                  PhotonDuplicates::V0Y,
                  PhotonDuplicates::V0Eta,                  
                  PhotonDuplicates::V0PosTrackTime,
                  PhotonDuplicates::V0NegTrackTime,
                  PhotonDuplicates::V0CollTime, 
                  PhotonDuplicates::V0NDuplicates,                  
                  PhotonDuplicates::V0GroupID,
                  PhotonDuplicates::V0PDGCode,                                    
                  PhotonDuplicates::IsCorrectlyAssoc);            
} // namespace o2::aod

#endif // PWGLF_DATAMODEL_LFSTRANGENESSMLTABLES_H_
