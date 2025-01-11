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
//
// This is a task that reads sigma0 tables (from sigma0builder) to perform analysis.
//  *+-+*+-+*+-+*+-+*+-+*+-+*+-+*+-+*
//  Sigma0 QA task
//  *+-+*+-+*+-+*+-+*+-+*+-+*+-+*+-+*
//
//    Comments, questions, complaints, suggestions?
//    Please write to:
//    gianni.shigeru.setoue.liveraro@cern.ch
//

#include <Math/Vector4D.h>
#include <cmath>
#include <array>
#include <cstdlib>

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "Framework/ASoA.h"
#include "ReconstructionDataFormats/Track.h"
#include "Common/Core/RecoDecay.h"
#include "Common/Core/trackUtilities.h"
#include "PWGLF/DataModel/LFStrangenessTables.h"
#include "PWGLF/DataModel/LFStrangenessPIDTables.h"
#include "PWGLF/DataModel/LFStrangenessMLTables.h"
#include "PWGLF/DataModel/LFSigmaTables.h"
#include "Common/Core/TrackSelection.h"
#include "Common/DataModel/TrackSelectionTables.h"
#include "Common/DataModel/EventSelection.h"
#include "Common/DataModel/Centrality.h"
#include "Common/DataModel/PIDResponse.h"
#include "CCDB/BasicCCDBManager.h"
#include <TFile.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TLorentzVector.h>
#include <TPDGCode.h>
#include <TDatabasePDG.h>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

using V0MCSigmas = soa::Join<aod::Sigma0Cores, aod::SigmaPhotonExtras, aod::SigmaLambdaExtras, aod::SigmaMCCores>;
using V0Sigmas = soa::Join<aod::Sigma0Cores, aod::SigmaPhotonExtras, aod::SigmaLambdaExtras>;

struct sigma0QA {
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  Configurable<bool> fProcessMonteCarlo{"fProcessMonteCarlo", true, "Flag to process MC data."};

  // For Standard Selection:
  //// Lambda standard criteria::
  Configurable<float> LambdaMinDCANegToPv{"LambdaMinDCANegToPv", .05, "min DCA Neg To PV (cm)"};
  Configurable<float> LambdaMinDCAPosToPv{"LambdaMinDCAPosToPv", .05, "min DCA Pos To PV (cm)"};
  Configurable<float> LambdaMaxDCAV0Dau{"LambdaMaxDCAV0Dau", 2.5, "Max DCA V0 Daughters (cm)"};
  Configurable<float> LambdaMinv0radius{"LambdaMinv0radius", 0.0, "Min V0 radius (cm)"};
  Configurable<float> LambdaMaxv0radius{"LambdaMaxv0radius", 40, "Max V0 radius (cm)"};
  Configurable<float> LambdaMinQt{"LambdaMinQt", 0.01, "Min lambda qt value (AP plot) (GeV/c)"};
  Configurable<float> LambdaMaxQt{"LambdaMaxQt", 0.17, "Max lambda qt value (AP plot) (GeV/c)"};
  Configurable<float> LambdaMinAlpha{"LambdaMinAlpha", 0.25, "Min lambda alpha absolute value (AP plot)"};
  Configurable<float> LambdaMaxAlpha{"LambdaMaxAlpha", 1.0, "Max lambda alpha absolute value (AP plot)"};
  Configurable<float> LambdaMinv0cospa{"LambdaMinv0cospa", 0.95, "Min V0 CosPA"};
  Configurable<float> LambdaWindow{"LambdaWindow", 0.015, "Mass window around expected (in GeV/c2)"};
  Configurable<float> LambdaMaxRap{"LambdaMaxRap", 0.8, "Max lambda rapidity"};
  Configurable<float> LambdaMaxTPCNSigmas{"LambdaMaxTPCNSigmas", 1e+9, "Max TPC NSigmas for daughters"};
  Configurable<float> LambdaMaxTOFNSigmas{"LambdaMaxTOFNSigmas", 1e+9, "Max TOF NSigmas for daughters"};

  //// Photon standard criteria:
  Configurable<float> PhotonDauMinPt{"PhotonDauMinPt", 0.0, "Min daughter pT (GeV/c)"};
  Configurable<float> PhotonMinDCADauToPv{"PhotonMinDCADauToPv", 0.0, "Min DCA daughter To PV (cm)"};
  Configurable<float> PhotonMaxDCAV0Dau{"PhotonMaxDCAV0Dau", 3.5, "Max DCA V0 Daughters (cm)"};
  Configurable<float> PhotonMinTPCCrossedRows{"PhotonMinTPCCrossedRows", 0, "Min daughter TPC Crossed Rows"};
  Configurable<float> PhotonMinTPCNSigmas{"PhotonMinTPCNSigmas", -7, "Min TPC NSigmas for daughters"};
  Configurable<float> PhotonMaxTPCNSigmas{"PhotonMaxTPCNSigmas", 7, "Max TPC NSigmas for daughters"};
  Configurable<float> PhotonMinPt{"PhotonMinPt", 0.0, "Min photon pT (GeV/c)"};
  Configurable<float> PhotonMaxPt{"PhotonMaxPt", 50.0, "Max photon pT (GeV/c)"};
  Configurable<float> PhotonMaxRap{"PhotonMaxRap", 0.5, "Max photon rapidity"};
  Configurable<float> PhotonMinRadius{"PhotonMinRadius", 3.0, "Min photon conversion radius (cm)"};
  Configurable<float> PhotonMaxRadius{"PhotonMaxRadius", 115, "Max photon conversion radius (cm)"};
  Configurable<float> PhotonMaxZ{"PhotonMaxZ", 240, "Max photon conversion point z value (cm)"};
  Configurable<float> PhotonMaxQt{"PhotonMaxQt", 0.05, "Max photon qt value (AP plot) (GeV/c)"};
  Configurable<float> PhotonMaxAlpha{"PhotonMaxAlpha", 0.95, "Max photon alpha absolute value (AP plot)"};
  Configurable<float> PhotonMinV0cospa{"PhotonMinV0cospa", 0.80, "Min V0 CosPA"};
  Configurable<float> PhotonMaxMass{"PhotonMaxMass", 0.10, "Max photon mass (GeV/c^{2})"};
  // TODO: Include PsiPair selection

  Configurable<float> SigmaMaxRap{"SigmaMaxRap", 0.5, "Max sigma0 rapidity"};

  // Axis
  // base properties
  ConfigurableAxis axisCentrality{"axisCentrality", {VARIABLE_WIDTH, 0.0f, 5.0f, 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 90.0f, 100.0f, 110.0f}, "Centrality"};
  ConfigurableAxis axisPt{"axisPt", {VARIABLE_WIDTH, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.2f, 2.4f, 2.6f, 2.8f, 3.0f, 3.2f, 3.4f, 3.6f, 3.8f, 4.0f, 4.4f, 4.8f, 5.2f, 5.6f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 17.0f, 19.0f, 21.0f, 23.0f, 25.0f, 30.0f, 35.0f, 40.0f, 50.0f}, "p_{T} (GeV/c)"};
  ConfigurableAxis axisRapidity{"axisRapidity", {100, -2.0f, 2.0f}, "Rapidity"};

  // Invariant Mass
  ConfigurableAxis axisSigmaMass{"axisSigmaMass", {500, 1.10f, 1.30f}, "M_{#Sigma^{0}} (GeV/c^{2})"};
  ConfigurableAxis axisLambdaMass{"axisLambdaMass", {200, 0.0f, 0.1f}, "M_{#Lambda} (GeV/c^{2})"};
  ConfigurableAxis axisPhotonMass{"axisPhotonMass", {500, 0.0f, 0.5f}, "M_{#Gamma}"};

  // AP plot axes
  ConfigurableAxis axisAPAlpha{"axisAPAlpha", {220, 0.0f, 1.1f}, "V0 AP alpha"};
  ConfigurableAxis axisAPQt{"axisAPQt", {220, 0.0f, 0.5f}, "V0 AP alpha"};

  // Track quality axes
  ConfigurableAxis axisTPCrows{"axisTPCrows", {160, 0.0f, 160.0f}, "N TPC rows"};

  // topological variable QA axes
  ConfigurableAxis axisRadius{"axisRadius", {240, 0.0f, 120.0f}, "V0 radius (cm)"};
  ConfigurableAxis axisDCAtoPV{"axisDCAtoPV", {500, 0.0f, 50.0f}, "DCA (cm)"};
  ConfigurableAxis axisDCAdau{"axisDCAdau", {50, 0.0f, 5.0f}, "DCA (cm)"};
  ConfigurableAxis axisCosPA{"axisCosPA", {400, 0.0f, 2.0f}, "Cosine of pointing angle"};
  ConfigurableAxis axisCandSel{"axisCandSel", {26, 0.5f, +26.5f}, "Candidate Selection"};

  int nSigmaCandidates = 0;
  void init(InitContext const&)
  {
    // All candidates received
    histos.add("h3dphotonMass", "h3dphotonMass", kTH3F, {axisPt, axisSigmaMass, axisPhotonMass});
    histos.add("h3dphotonMinDCADauToPv", "h3dphotonMinDCADauToPv", kTH3F, {axisPt, axisSigmaMass, axisDCAtoPV});
    histos.add("h3dphotonDCADau", "h3dphotonDCADau", kTH3F, {axisPt, axisSigmaMass, axisDCAdau});
    histos.add("h3dphotonRadius", "h3dphotonRadius", kTH3F, {axisPt, axisSigmaMass, axisRadius});
    histos.add("h3dphotonZconv", "h3dphotonZconv", kTH3F, {axisPt, axisSigmaMass, {240, 0.0f, 120.0f}});
    histos.add("h3dphotonQt", "h3dphotonQt", kTH3F, {axisPt, axisSigmaMass, axisAPQt});
    histos.add("h3dphotonAlpha", "h3dphotonAlpha", kTH3F, {axisPt, axisSigmaMass, axisAPAlpha});
    histos.add("h3dphotonPA", "h3dphotonPA", kTH3F, {axisPt, axisSigmaMass, axisCosPA});
    histos.add("h3dlambdaMass", "h3dlambdaMass", kTH3F, {axisPt, axisSigmaMass, axisLambdaMass});
    histos.add("h3dlambdaMinDCAToPv", "h3dlambdaMinDCAToPv", kTH3F, {axisPt, axisSigmaMass, axisDCAtoPV});
    histos.add("h3dlambdaRadius", "h3dlambdaRadius", kTH3F, {axisPt, axisSigmaMass, axisRadius});
    histos.add("h3dlambdaDCADau", "h3dlambdaDCADau", kTH3F, {axisPt, axisSigmaMass, axisDCAdau});
    histos.add("h3dlambdaQt", "h3dlambdaQt", kTH3F, {axisPt, axisSigmaMass, axisAPQt});
    histos.add("h3dlambdaAlpha", "h3dlambdaAlpha", kTH3F, {axisPt, axisSigmaMass, axisAPAlpha});
    histos.add("h3dlambdaPA", "h3dlambdaPA", kTH3F, {axisPt, axisSigmaMass, axisCosPA});
    histos.add("hSigmaMass", "hSigmaMass", kTH1F, {axisSigmaMass});

    if (fProcessMonteCarlo) {
        histos.add("h3dphotonMass_Signal", "h3dphotonMass_Signal", kTH3F, {axisPt, axisSigmaMass, axisPhotonMass});
        histos.add("h3dphotonMinDCADauToPv_Signal", "h3dphotonMinDCADauToPv_Signal", kTH3F, {axisPt, axisSigmaMass, axisDCAtoPV});
        histos.add("h3dphotonDCADau_Signal", "h3dphotonDCADau_Signal", kTH3F, {axisPt, axisSigmaMass, axisDCAdau});
        histos.add("h3dphotonRadius_Signal", "h3dphotonRadius_Signal", kTH3F, {axisPt, axisSigmaMass, axisRadius});
        histos.add("h3dphotonZconv_Signal", "h3dphotonZconv_Signal", kTH3F, {axisPt, axisSigmaMass, {240, 0.0f, 120.0f}});
        histos.add("h3dphotonQt_Signal", "h3dphotonQt_Signal", kTH3F, {axisPt, axisSigmaMass, axisAPQt});
        histos.add("h3dphotonAlpha_Signal", "h3dphotonAlpha_Signal", kTH3F, {axisPt, axisSigmaMass, axisAPAlpha});
        histos.add("h3dphotonPA_Signal", "h3dphotonPA_Signal", kTH3F, {axisPt, axisSigmaMass, axisCosPA});
        histos.add("h3dlambdaMass_Signal", "h3dlambdaMass_Signal", kTH3F, {axisPt, axisSigmaMass, axisLambdaMass});
        histos.add("h3dlambdaMinDCAToPv_Signal", "h3dlambdaMinDCAToPv_Signal", kTH3F, {axisPt, axisSigmaMass, axisDCAtoPV});
        histos.add("h3dlambdaRadius_Signal", "h3dlambdaRadius_Signal", kTH3F, {axisPt, axisSigmaMass, axisRadius});
        histos.add("h3dlambdaDCADau_Signal", "h3dlambdaDCADau_Signal", kTH3F, {axisPt, axisSigmaMass, axisDCAdau});
        histos.add("h3dlambdaQt_Signal", "h3dlambdaQt_Signal", kTH3F, {axisPt, axisSigmaMass, axisAPQt});
        histos.add("h3dlambdaAlpha_Signal", "h3dlambdaAlpha_Signal", kTH3F, {axisPt, axisSigmaMass, axisAPAlpha});
        histos.add("h3dlambdaPA_Signal", "h3dlambdaPA_Signal", kTH3F, {axisPt, axisSigmaMass, axisCosPA});
        histos.add("hSigmaMass_Signal", "hSigmaMass_Signal", kTH1F, {axisSigmaMass});
    }
  }

  // Apply selections in sigma candidates
  template <typename TV0Object>
  bool processSigmaCandidate(TV0Object const& cand, int selection)
  {
    if ((TMath::Abs(cand.photonMass()) > PhotonMaxMass) && (selection!=0))
        return false;
    if ((cand.photonPosPt() < PhotonDauMinPt) || (cand.photonNegPt() < PhotonDauMinPt))
        return false;
    if (((TMath::Abs(cand.photonDCAPosPV()) < PhotonMinDCADauToPv) || (TMath::Abs(cand.photonDCANegPV()) < PhotonMinDCADauToPv)) && (selection!=1))
        return false;
    if ((TMath::Abs(cand.photonDCADau()) > PhotonMaxDCAV0Dau) && (selection!=2))
        return false;
    if ((cand.photonPosTPCCrossedRows() < PhotonMinTPCCrossedRows) || (cand.photonNegTPCCrossedRows() < PhotonMinTPCCrossedRows))
        return false;
    if ((cand.photonPosTPCNSigma() != -999.f) && ((cand.photonPosTPCNSigma() < PhotonMinTPCNSigmas) || (cand.photonPosTPCNSigma() > PhotonMaxTPCNSigmas)))
        return false;
    if ((cand.photonNegTPCNSigma() != -999.f) && ((cand.photonNegTPCNSigma() < PhotonMinTPCNSigmas) || (cand.photonNegTPCNSigma() > PhotonMaxTPCNSigmas)))
        return false;
    if ((cand.photonPt() < PhotonMinPt) || (cand.photonPt() > PhotonMaxPt))
        return false;
    if ((TMath::Abs(cand.photonY()) > PhotonMaxRap))
        return false;
    if (((cand.photonRadius() < PhotonMinRadius) || (cand.photonRadius() > PhotonMaxRadius)) && (selection!=3))
        return false;
    if ((TMath::Abs(cand.photonZconv()) > PhotonMaxZ) && (selection!=4))
        return false;
    if ((cand.photonQt() > PhotonMaxQt) && (selection!=5))
        return false;
    if ((TMath::Abs(cand.photonAlpha()) > PhotonMaxAlpha) && (selection!=6))
        return false;
    if ((cand.photonCosPA() < PhotonMinV0cospa) && (selection!=7))
        return false;

    // Lambda selection
    if ((TMath::Abs(cand.lambdaMass() - 1.115683) > LambdaWindow) && (TMath::Abs(cand.antilambdaMass() - 1.115683) > LambdaWindow) && (selection!=8))
        return false;
    if (((TMath::Abs(cand.lambdaDCAPosPV()) < LambdaMinDCAPosToPv) || (TMath::Abs(cand.lambdaDCANegPV()) < LambdaMinDCANegToPv)) && (selection!=9))
        return false;
    if (((cand.lambdaRadius() < LambdaMinv0radius) || (cand.lambdaRadius() > LambdaMaxv0radius)) && (selection!=10))
        return false;
    if ((TMath::Abs(cand.lambdaDCADau()) > LambdaMaxDCAV0Dau) && (selection!=11))
        return false;
    if (((cand.lambdaQt() < LambdaMinQt) || (cand.lambdaQt() > LambdaMaxQt)) && (selection!=12))
        return false;
    if (((TMath::Abs(cand.lambdaAlpha()) < LambdaMinAlpha) || (TMath::Abs(cand.lambdaAlpha()) > LambdaMaxAlpha)) && (selection!=13))
        return false;
    if ((cand.lambdaCosPA() < LambdaMinv0cospa) && (selection!=14))
        return false;
    if ((TMath::Abs(cand.lambdaY()) > LambdaMaxRap))
        return false;
    if (TMath::Abs(cand.sigmaRapidity()) > SigmaMaxRap)
        return false;

    return true;
  }

  void processMonteCarlo(V0MCSigmas const& v0s)
  {
    for (auto& sigma : v0s) { // selecting Sigma0-like candidates

        bool IsSigmaLike = (sigma.isSigma() || sigma.isAntiSigma()); 
        float SigmapT = sigma.sigmapT();
        float SigmaMass = sigma.sigmaMass();
        
        if (IsSigmaLike){
            histos.fill(HIST("hSigmaMass_Signal"), SigmaMass);
            if (processSigmaCandidate(sigma,0))
                histos.fill(HIST("h3dphotonMass_Signal"), SigmapT, SigmaMass, TMath::Abs(sigma.photonMass()));
            if (processSigmaCandidate(sigma,1)){
                histos.fill(HIST("h3dphotonMinDCADauToPv_Signal"), SigmapT, SigmaMass, TMath::Abs(sigma.photonDCAPosPV()));
                histos.fill(HIST("h3dphotonMinDCADauToPv_Signal"), SigmapT, SigmaMass, TMath::Abs(sigma.photonDCANegPV()));
            }
            if (processSigmaCandidate(sigma,2))
                histos.fill(HIST("h3dphotonDCADau_Signal"), SigmapT, SigmaMass, TMath::Abs(sigma.photonDCADau()));
            if (processSigmaCandidate(sigma,3))
                histos.fill(HIST("h3dphotonRadius_Signal"), SigmapT, SigmaMass, sigma.photonRadius());
            if (processSigmaCandidate(sigma,4))
                histos.fill(HIST("h3dphotonZconv_Signal"), SigmapT, SigmaMass, sigma.photonZconv());
            if (processSigmaCandidate(sigma,5))
                histos.fill(HIST("h3dphotonQt_Signal"), SigmapT, SigmaMass, sigma.photonQt());
            if (processSigmaCandidate(sigma,6))
                histos.fill(HIST("h3dphotonAlpha_Signal"), SigmapT, SigmaMass, TMath::Abs(sigma.photonAlpha()));
            if (processSigmaCandidate(sigma,7))
                histos.fill(HIST("h3dphotonPA_Signal"), SigmapT, SigmaMass, TMath::ACos(sigma.photonCosPA()));
            if (processSigmaCandidate(sigma,8))
                histos.fill(HIST("h3dlambdaMass_Signal"), SigmapT, SigmaMass, TMath::Abs(sigma.lambdaMass() - 1.115683));
            if (processSigmaCandidate(sigma,9)){
                histos.fill(HIST("h3dlambdaMinDCAToPv_Signal"), SigmapT, SigmaMass, TMath::Abs(sigma.lambdaDCAPosPV()));
                histos.fill(HIST("h3dlambdaMinDCAToPv_Signal"), SigmapT, SigmaMass, TMath::Abs(sigma.lambdaDCANegPV()));
            }
            if (processSigmaCandidate(sigma,10))
                histos.fill(HIST("h3dlambdaRadius_Signal"), SigmapT, SigmaMass, sigma.lambdaRadius());
            if (processSigmaCandidate(sigma,11))
                histos.fill(HIST("h3dlambdaDCADau_Signal"), SigmapT, SigmaMass, TMath::Abs(sigma.lambdaDCADau()));
            if (processSigmaCandidate(sigma,12))
                histos.fill(HIST("h3dlambdaQt_Signal"), SigmapT, SigmaMass, sigma.lambdaQt());
            if (processSigmaCandidate(sigma,13))
                histos.fill(HIST("h3dlambdaAlpha_Signal"), SigmapT, SigmaMass, TMath::Abs(sigma.lambdaAlpha()));
            if (processSigmaCandidate(sigma,14))
                histos.fill(HIST("h3dlambdaPA_Signal"), SigmapT, SigmaMass, TMath::ACos(sigma.lambdaCosPA()));
        }
        else{
            histos.fill(HIST("hSigmaMass"), SigmaMass);
            if (processSigmaCandidate(sigma,0))
                histos.fill(HIST("h3dphotonMass"), SigmapT, SigmaMass, TMath::Abs(sigma.photonMass()));
            if (processSigmaCandidate(sigma,1)){
                histos.fill(HIST("h3dphotonMinDCADauToPv"), SigmapT, SigmaMass, TMath::Abs(sigma.photonDCAPosPV()));
                histos.fill(HIST("h3dphotonMinDCADauToPv"), SigmapT, SigmaMass, TMath::Abs(sigma.photonDCANegPV()));
            }
            if (processSigmaCandidate(sigma,2))
                histos.fill(HIST("h3dphotonDCADau"), SigmapT, SigmaMass, TMath::Abs(sigma.photonDCADau()));
            if (processSigmaCandidate(sigma,3))
                histos.fill(HIST("h3dphotonRadius"), SigmapT, SigmaMass, sigma.photonRadius());
            if (processSigmaCandidate(sigma,4))
                histos.fill(HIST("h3dphotonZconv"), SigmapT, SigmaMass, sigma.photonZconv());
            if (processSigmaCandidate(sigma,5))
                histos.fill(HIST("h3dphotonQt"), SigmapT, SigmaMass, sigma.photonQt());
            if (processSigmaCandidate(sigma,6))
                histos.fill(HIST("h3dphotonAlpha"), SigmapT, SigmaMass, TMath::Abs(sigma.photonAlpha()));
            if (processSigmaCandidate(sigma,7))
                histos.fill(HIST("h3dphotonPA"), SigmapT, SigmaMass, TMath::ACos(sigma.photonCosPA()));
            if (processSigmaCandidate(sigma,8))
                histos.fill(HIST("h3dlambdaMass"), SigmapT, SigmaMass, TMath::Abs(sigma.lambdaMass() - 1.115683));
            if (processSigmaCandidate(sigma,9)){
                histos.fill(HIST("h3dlambdaMinDCAToPv"), SigmapT, SigmaMass, TMath::Abs(sigma.lambdaDCAPosPV()));
                histos.fill(HIST("h3dlambdaMinDCAToPv"), SigmapT, SigmaMass, TMath::Abs(sigma.lambdaDCANegPV()));
            }
            if (processSigmaCandidate(sigma,10))
                histos.fill(HIST("h3dlambdaRadius"), SigmapT, SigmaMass, sigma.lambdaRadius());
            if (processSigmaCandidate(sigma,11))
                histos.fill(HIST("h3dlambdaDCADau"), SigmapT, SigmaMass, TMath::Abs(sigma.lambdaDCADau()));
            if (processSigmaCandidate(sigma,12))
                histos.fill(HIST("h3dlambdaQt"), SigmapT, SigmaMass, sigma.lambdaQt());
            if (processSigmaCandidate(sigma,13))
                histos.fill(HIST("h3dlambdaAlpha"), SigmapT, SigmaMass, TMath::Abs(sigma.lambdaAlpha()));
            if (processSigmaCandidate(sigma,14))
                histos.fill(HIST("h3dlambdaPA"), SigmapT, SigmaMass, TMath::ACos(sigma.lambdaCosPA()));
        }
        
    }
  }

  void processRealData(V0Sigmas const& v0s)
  {
    for (auto& sigma : v0s) { // selecting Sigma0-like candidates
      
        nSigmaCandidates++;
        if (nSigmaCandidates % 50000 == 0) {
        LOG(info) << "Sigma0 Candidates processed: " << nSigmaCandidates;
        }
    
        bool IsSigmaLike = false; 
        float SigmapT = sigma.sigmapT();
        float SigmaMass = sigma.sigmaMass();

        if (processSigmaCandidate(sigma,0))
            histos.fill(HIST("h3dphotonMass"), SigmapT, SigmaMass, TMath::Abs(sigma.photonMass()));
        if (processSigmaCandidate(sigma,1)){
            histos.fill(HIST("h3dphotonMinDCADauToPv"), SigmapT, SigmaMass, TMath::Abs(sigma.photonDCAPosPV()));
            histos.fill(HIST("h3dphotonMinDCADauToPv"), SigmapT, SigmaMass, TMath::Abs(sigma.photonDCANegPV()));
        }
        if (processSigmaCandidate(sigma,2))
            histos.fill(HIST("h3dphotonDCADau"), SigmapT, SigmaMass, TMath::Abs(sigma.photonDCADau()));
        if (processSigmaCandidate(sigma,3))
            histos.fill(HIST("h3dphotonRadius"), SigmapT, SigmaMass, sigma.photonRadius());
        if (processSigmaCandidate(sigma,4))
            histos.fill(HIST("h3dphotonZconv"), SigmapT, SigmaMass, sigma.photonZconv());
        if (processSigmaCandidate(sigma,5))
            histos.fill(HIST("h3dphotonQt"), SigmapT, SigmaMass, sigma.photonQt());
        if (processSigmaCandidate(sigma,6))
            histos.fill(HIST("h3dphotonAlpha"), SigmapT, SigmaMass, TMath::Abs(sigma.photonAlpha()));
        if (processSigmaCandidate(sigma,7))
            histos.fill(HIST("h3dphotonPA"), SigmapT, SigmaMass, TMath::ACos(sigma.photonCosPA()));
        if (processSigmaCandidate(sigma,8))
            histos.fill(HIST("h3dlambdaMass"), SigmapT, SigmaMass, TMath::Abs(sigma.lambdaMass() - 1.115683));
        if (processSigmaCandidate(sigma,9)){
            histos.fill(HIST("h3dlambdaMinDCAToPv"), SigmapT, SigmaMass, TMath::Abs(sigma.lambdaDCAPosPV()));
            histos.fill(HIST("h3dlambdaMinDCAToPv"), SigmapT, SigmaMass, TMath::Abs(sigma.lambdaDCANegPV()));
        }
        if (processSigmaCandidate(sigma,10))
            histos.fill(HIST("h3dlambdaRadius"), SigmapT, SigmaMass, sigma.lambdaRadius());
        if (processSigmaCandidate(sigma,11))
            histos.fill(HIST("h3dlambdaDCADau"), SigmapT, SigmaMass, TMath::Abs(sigma.lambdaDCADau()));
        if (processSigmaCandidate(sigma,12))
            histos.fill(HIST("h3dlambdaQt"), SigmapT, SigmaMass, sigma.lambdaQt());
        if (processSigmaCandidate(sigma,13))
            histos.fill(HIST("h3dlambdaAlpha"), SigmapT, SigmaMass, TMath::Abs(sigma.lambdaAlpha()));
        if (processSigmaCandidate(sigma,14))
            histos.fill(HIST("h3dlambdaPA"), SigmapT, SigmaMass, TMath::ACos(sigma.lambdaCosPA()));
        
        histos.fill(HIST("hSigmaMass"), SigmaMass);
    }
  }

  PROCESS_SWITCH(sigma0QA, processMonteCarlo, "Do Monte-Carlo-based analysis", true);
  PROCESS_SWITCH(sigma0QA, processRealData, "Do real data analysis", false);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{adaptAnalysisTask<sigma0QA>(cfgc)};
}

